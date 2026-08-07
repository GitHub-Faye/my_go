//! Kaya deadstones 原生内核 —— PyO3 绑定。
//!
//! 逐函数移植自 `packages/deadstones/src-rust/src/{rand.rs,pseudo_board.rs,deadstones.rs}`，
//! 保持算法与语义完全一致。与 wasm 版唯一的差异：
//!   - 输入用 numpy 2D 数组 (H,W) int8 而非扁平 Vec<Sign>
//!   - 经 numpy crate 桥接，内部仍是扁平 Vec<Sign>（局部性/缓存友好）
//! 这样性能与原生 Rust 一样（`opt-level=3` + `lto`），远快于纯 Python 逐点操作。
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

type Sign = i8;

// ── rand.rs ──────────────────────────────────────────────

const KX: u32 = 123456789;
const KY: u32 = 362436069;
const KZ: u32 = 521288629;
const KW: u32 = 88675123;

struct Rand {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

impl Rand {
    fn new(seed: u32) -> Rand {
        Rand {
            x: KX ^ seed,
            y: KY ^ seed,
            z: KZ,
            w: KW,
        }
    }

    // Xorshift 128
    fn rand(&mut self) -> u32 {
        let t = self.x ^ self.x.wrapping_shl(11);

        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        self.w ^= self.w.wrapping_shr(19) ^ t ^ t.wrapping_shr(8);

        self.w
    }

    fn range(&mut self, a: i32, b: i32) -> i32 {
        let m = (b - a) as u32;
        a + self.rand().checked_rem(m).unwrap_or(0) as i32
    }
}

// ── pseudo_board.rs ──────────────────────────────────────

type Vertex = usize;

#[derive(Clone)]
struct PseudoBoard {
    data: Vec<Sign>,
    width: usize,
}

impl PseudoBoard {
    fn get(&self, v: Vertex) -> Option<Sign> {
        self.data.get(v).cloned()
    }

    fn set(&mut self, v: Vertex, sign: Sign) {
        if let Some(x) = self.data.get_mut(v) {
            *x = sign;
        }
    }

    fn get_neighbors(&self, v: Vertex) -> Vec<Vertex> {
        let mut result = vec![v.wrapping_sub(self.width), v + self.width];
        let x = v % self.width;

        if x > 0 {
            result.push(v - 1);
        }

        if x < self.width - 1 {
            result.push(v + 1);
        }

        result
    }

    fn get_connected_component_inner(&self, vertex: Vertex, signs: &[Sign], result: &mut Vec<Vertex>) {
        result.push(vertex);

        for neighbor in self.get_neighbors(vertex).into_iter() {
            let s = match self.get(neighbor) {
                Some(x) => x,
                None => continue,
            };

            if signs.contains(&s) && !result.contains(&neighbor) {
                self.get_connected_component_inner(neighbor, signs, result);
            }
        }
    }

    fn get_connected_component(&self, vertex: Vertex, signs: &[Sign]) -> Vec<Vertex> {
        let mut result = vec![];
        self.get_connected_component_inner(vertex, signs, &mut result);
        result
    }

    fn get_related_chains(&self, vertex: Vertex) -> Vec<Vertex> {
        let sign = match self.get(vertex) {
            Some(x) => x,
            None => return vec![],
        };

        self.get_connected_component(vertex, &[sign, 0])
            .into_iter()
            .filter(|&v| self.get(v) == Some(sign))
            .collect()
    }

    fn get_chain(&self, vertex: Vertex) -> Vec<Vertex> {
        let sign = match self.get(vertex) {
            Some(x) => x,
            None => return vec![],
        };

        self.get_connected_component(vertex, &[sign])
    }

    fn has_liberties_inner(&self, vertex: Vertex, visited: &mut Vec<Vertex>, sign: Sign) -> bool {
        visited.push(vertex);

        for neighbor in self.get_neighbors(vertex).into_iter() {
            match self.get(neighbor) {
                Some(0) => return true,
                Some(x) if x == sign && !visited.contains(&neighbor) => {
                    if self.has_liberties_inner(neighbor, visited, sign) {
                        return true;
                    }
                }
                _ => {}
            }
        }

        false
    }

    fn has_liberties(&self, vertex: Vertex) -> bool {
        self.has_liberties_inner(
            vertex,
            &mut vec![],
            match self.get(vertex) {
                Some(x) => x,
                None => return false,
            },
        )
    }

    fn make_pseudo_move(&mut self, sign: Sign, vertex: Vertex) -> Option<Vec<Vertex>> {
        let neighbors = self.get_neighbors(vertex);
        let mut check_capture = false;
        let mut check_multi_dead_chains = false;

        if neighbors.iter().all(|&neighbor| {
            let s = self.get(neighbor);
            s == None || s == Some(sign)
        }) {
            return None;
        }

        self.set(vertex, sign);

        if !self.has_liberties(vertex) {
            let is_point_chain = neighbors.iter().all(|&n| self.get(n) != Some(sign));

            if is_point_chain {
                check_multi_dead_chains = true;
            } else {
                check_capture = true;
            }
        }

        let mut dead = vec![];
        let mut dead_chains = 0;

        for neighbor in neighbors.into_iter() {
            if self.get(neighbor) != Some(-sign) || self.has_liberties(neighbor) {
                continue;
            }

            let chain = self.get_chain(neighbor);
            dead_chains += 1;

            for c in chain.into_iter() {
                self.set(c, 0);
                dead.push(c);
            }
        }

        if check_multi_dead_chains && dead_chains <= 1 || check_capture && dead.len() == 0 {
            for &d in &dead {
                self.set(d, -sign);
            }

            self.set(vertex, 0);
            return None;
        }

        Some(dead)
    }

    fn get_floating_stones(&self) -> Vec<Vertex> {
        let mut done = vec![];
        let mut result = vec![];

        for vertex in 0..self.data.len() {
            if self.get(vertex) != Some(0) || done.contains(&vertex) {
                continue;
            }

            let pos_area = self.get_connected_component(vertex, &[0, -1]);
            let neg_area = self.get_connected_component(vertex, &[0, 1]);
            let pos_dead = pos_area
                .iter()
                .cloned()
                .filter(|&v| self.get(v) == Some(-1))
                .collect::<Vec<_>>();
            let neg_dead = neg_area
                .iter()
                .cloned()
                .filter(|&v| self.get(v) == Some(1))
                .collect::<Vec<_>>();
            let pos_diff = pos_area
                .iter()
                .filter(|&v| !pos_dead.contains(v) && !neg_area.contains(v))
                .count();
            let neg_diff = neg_area
                .iter()
                .filter(|&v| !neg_dead.contains(v) && !pos_area.contains(v))
                .count();

            let favor_neg = neg_diff <= 1 && neg_dead.len() <= pos_dead.len();
            let favor_pos = pos_diff <= 1 && pos_dead.len() <= neg_dead.len();

            let (mut actual_area, mut actual_dead) = match (favor_neg, favor_pos) {
                (false, true) => (pos_area, pos_dead),
                (true, false) => (neg_area, neg_dead),
                _ => (self.get_chain(vertex), vec![]),
            };

            done.append(&mut actual_area);
            result.append(&mut actual_dead);
        }

        result
    }
}

// ── deadstones.rs（只移植 get_probability_map / play_till_end，guess 由 Python 层组合）──

fn get_probability_map_core(board: &PseudoBoard, iterations: usize, rand: &mut Rand) -> Vec<f32> {
    let mut result = board.data.iter().map(|_| (0u32, 0u32)).collect::<Vec<_>>();

    for i in 0..iterations {
        let sign = if i < iterations / 2 { -1 } else { 1 };
        let area_map = play_till_end_core(board.clone(), sign, rand);

        for v in 0..area_map.data.len() {
            let s = match area_map.get(v) {
                Some(x) => x,
                None => continue,
            };

            if let Some(slots) = result.get_mut(v) {
                if s == -1 {
                    slots.0 += 1;
                } else if s == 1 {
                    slots.1 += 1;
                }
            }
        }
    }

    result
        .into_iter()
        .map(|(n, p)| match p + n {
            0 => 0.0,
            _ => p as f32 * 2.0 / (p + n) as f32 - 1.0,
        })
        .collect()
}

fn play_till_end_core(mut board: PseudoBoard, mut sign: Sign, rand: &mut Rand) -> PseudoBoard {
    let mut illegal_vertices = vec![];
    let mut finished = (false, false);
    let mut free_vertices = (0..board.data.len())
        .filter(|&v| board.get(v) == Some(0))
        .collect::<Vec<_>>();

    while free_vertices.len() > 0 && (!finished.0 || !finished.1) {
        let mut made_move = false;

        while free_vertices.len() > 0 {
            let random_index = rand.range(0, free_vertices.len() as i32) as usize;
            let vertex = *free_vertices.get(random_index).unwrap_or(&0);

            free_vertices.remove(random_index);

            if let Some(mut freed_vertices) = board.make_pseudo_move(sign, vertex) {
                free_vertices.append(&mut freed_vertices);

                if sign < 0 {
                    finished.0 = false;
                } else {
                    finished.1 = false;
                }

                made_move = true;
                break;
            } else {
                illegal_vertices.push(vertex);
            }
        }

        if sign > 0 {
            finished.0 = !made_move;
        } else {
            finished.1 = !made_move;
        }

        free_vertices.append(&mut illegal_vertices);
        sign = -sign;
    }

    // Patch holes

    for vertex in 0..board.data.len() {
        if board.get(vertex) != Some(0) {
            continue;
        }

        let mut sign = 0;

        for n in board.get_neighbors(vertex).into_iter() {
            let s = board.get(n);

            if s == Some(1) || s == Some(-1) {
                sign = s.unwrap_or(0);
                break;
            }
        }

        if sign != 0 {
            board.set(vertex, sign);
        }
    }

    board
}

// ── Python 绑定 ───────────────────────────────────────────

/// 从 numpy (H,W,int8) 数组构造扁平 Vec<Sign>（行优先）。
fn board_from_py(data: PyReadonlyArray2<'_, i8>) -> PseudoBoard {
    let arr = data.as_array();
    let height = arr.shape()[0];
    let width = arr.shape()[1];
    let mut v = Vec::with_capacity(height * width);
    for row in arr.rows() {
        for &s in row {
            v.push(s);
        }
    }
    PseudoBoard { data: v, width }
}

/// `getProbabilityMap(data, iterations, seed)` —— 领地概率图。
///
/// 对应 wasm 的 `getProbabilityMap`。data: (H,W) int8, 1=黑/-1=白/0=空
/// → (H,W) float32 ∈ [-1,1],正=黑控制、负=白控制。
#[pyfunction]
fn get_probability_map<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i8>,
    iterations: usize,
    seed: u32,
) -> Bound<'py, PyArray2<f32>> {
    let board = board_from_py(data);
    let map = crate::get_probability_map_impl(&board, iterations, &mut Rand::new(seed));
    let height = board.data.len() / board.width;
    numpy::ndarray::Array::from_shape_vec((height, board.width), map)
        .unwrap()
        .into_pyarray(py)
}

/// `playTillEnd(data, sign, seed)` —— 从满先手随机下到双方 pass。
#[pyfunction]
fn play_till_end<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i8>,
    sign: i8,
    seed: u32,
) -> Bound<'py, PyArray2<i8>> {
    let board = board_from_py(data);
    let result_board = crate::play_till_end_impl(board, sign, &mut Rand::new(seed));
    let height = result_board.data.len() / result_board.width;
    numpy::ndarray::Array::from_shape_vec((height, result_board.width), result_board.data)
        .unwrap()
        .into_pyarray(py)
}

/// `getFloatingStones(data)` —— 浮子（永不活着的孤立棋）。
#[pyfunction]
fn get_floating_stones(data: PyReadonlyArray2<'_, i8>) -> Vec<usize> {
    let board = board_from_py(data);
    board.get_floating_stones()
}

/// Python 模块入口。
#[pymodule]
fn kaya_deadstones_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_probability_map, m)?)?;
    m.add_function(wrap_pyfunction!(play_till_end, m)?)?;
    m.add_function(wrap_pyfunction!(get_floating_stones, m)?)?;
    Ok(())
}

/// 对应 deadstones.rs 的 get_probability_map。
fn get_probability_map_impl(board: &PseudoBoard, iterations: usize, rand: &mut Rand) -> Vec<f32> {
    get_probability_map_core(board, iterations, rand)
}

/// 对应 deadstones.rs 的 play_till_end。
fn play_till_end_impl(board: PseudoBoard, sign: Sign, rand: &mut Rand) -> PseudoBoard {
    play_till_end_core(board, sign, rand)
}
