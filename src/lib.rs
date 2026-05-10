use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut};
use std::ptr::NonNull;

// =============================================================================
// 维度模块 (dimension)
// =============================================================================

pub trait Dimension: Clone + std::fmt::Debug {
    fn ndim(&self) -> usize;
    fn shape(&self) -> &[usize];
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
    fn to_multi_index(&self, linear: usize) -> Vec<usize> {
        let shape = self.shape();
        let mut index = vec![0; shape.len()];
        let mut remaining = linear;
        for (dim, &dim_size) in index.iter_mut().zip(shape).rev() {
            *dim = remaining % dim_size;
            remaining /= dim_size;
        }
        index.reverse();
        index
    }
    fn to_linear(&self, indices: &[usize]) -> usize {
        let shape = self.shape();
        let mut linear = 0;
        let mut stride = 1;
        for (i, &dim_size) in shape.iter().enumerate().rev() {
            linear += indices[i] * stride;
            stride *= dim_size;
        }
        linear
    }
}

#[derive(Debug, Clone)]
pub struct Dim<const N: usize> {
    shape: [usize; N],
}

impl<const N: usize> Dim<N> {
    pub fn new(shape: [usize; N]) -> Self {
        Self { shape }
    }
}

impl<const N: usize> Dimension for Dim<N> {
    fn ndim(&self) -> usize {
        N
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[derive(Debug, Clone)]
pub struct IxDyn {
    shape: Vec<usize>,
}

impl IxDyn {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}

impl Dimension for IxDyn {
    fn ndim(&self) -> usize {
        self.shape.len()
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

// =============================================================================
// 布局模块 (layout)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
    ColMajor,
}

#[derive(Debug, Clone, Copy)]
pub struct LayoutRef<'a> {
    layout: Layout,
    strides: &'a [usize],
}

impl<'a> LayoutRef<'a> {
    pub fn new(layout: Layout, strides: &'a [usize]) -> Self {
        Self { layout, strides }
    }
    pub fn layout(&self) -> Layout {
        self.layout
    }
    pub fn strides(&self) -> &[usize] {
        self.strides
    }
}

// =============================================================================
// 数据表示模块 (repr)
// =============================================================================

pub unsafe trait Data {
    type Elem;
    fn ptr(&self) -> NonNull<Self::Elem>;
    fn len(&self) -> usize;
}

pub struct OwnedRepr<T> {
    data: Vec<T>,
}

impl<T> OwnedRepr<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T: Clone> Clone for OwnedRepr<T> {
    fn clone(&self) -> Self {
        OwnedRepr::new(self.data.clone())
    }
}

unsafe impl<T> Data for OwnedRepr<T> {
    type Elem = T;
    fn ptr(&self) -> NonNull<T> {
        NonNull::from(&self.data[0])
    }
    fn len(&self) -> usize {
        self.data.len()
    }
}

pub struct ViewRepr<T> {
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> ViewRepr<T> {
    pub unsafe fn new(ptr: NonNull<T>, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }
}

// 视图克隆只复制指针（浅拷贝）
impl<T> Clone for ViewRepr<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

unsafe impl<T> Data for ViewRepr<T> {
    type Elem = T;
    fn ptr(&self) -> NonNull<T> {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

// =============================================================================
// 数组核心结构 (array)
// =============================================================================

pub struct ArrayBase<S, D>
where
    S: Data,
    D: Dimension,
{
    data: S,
    dim: D,
    strides: Vec<usize>,
    layout: Layout,
}

impl<S, D> ArrayBase<S, D>
where
    S: Data,
    D: Dimension,
{
    pub fn new(data: S, dim: D, strides: Vec<usize>, layout: Layout) -> Self {
        Self {
            data,
            dim,
            strides,
            layout,
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.dim.shape()
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn len(&self) -> usize {
        self.dim.size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn linear_offset(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.dim.ndim(),
            "索引维度必须与数组维度匹配"
        );
        indices
            .iter()
            .zip(&self.strides)
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

    pub fn get(&self, indices: &[usize]) -> Option<&S::Elem> {
        if indices.iter().zip(self.dim.shape()).any(|(&idx, &dim)| idx >= dim) {
            return None;
        }
        let offset = self.linear_offset(indices);
        unsafe { self.data.ptr().as_ptr().add(offset).as_ref() }
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut S::Elem> {
        if indices.iter().zip(self.dim.shape()).any(|(&idx, &dim)| idx >= dim) {
            return None;
        }
        let offset = self.linear_offset(indices);
        unsafe { self.data.ptr().as_ptr().add(offset).as_mut() }
    }

    pub fn get_linear(&self, linear: usize) -> Option<&S::Elem> {
        if linear >= self.len() {
            return None;
        }
        unsafe { self.data.ptr().as_ptr().add(linear).as_ref() }
    }

    pub fn get_linear_mut(&mut self, linear: usize) -> Option<&mut S::Elem> {
        if linear >= self.len() {
            return None;
        }
        unsafe { self.data.ptr().as_ptr().add(linear).as_mut() }
    }

    pub fn view(&self) -> ArrayBase<ViewRepr<S::Elem>, D>
    where
        D: Clone,
    {
        unsafe {
            let ptr = self.data.ptr();
            let len = self.len();
            let view_data = ViewRepr::new(ptr, len);
            ArrayBase {
                data: view_data,
                dim: self.dim.clone(),
                strides: self.strides.clone(),
                layout: self.layout,
            }
        }
    }

    pub fn view_mut(&mut self) -> ArrayBase<ViewRepr<S::Elem>, D>
    where
        D: Clone,
    {
        unsafe {
            let ptr = self.data.ptr();
            let len = self.len();
            let view_data = ViewRepr::new(ptr, len);
            ArrayBase {
                data: view_data,
                dim: self.dim.clone(),
                strides: self.strides.clone(),
                layout: self.layout,
            }
        }
    }
}

// 克隆实现（要求元素可克隆，且数据表示可克隆）
impl<S, D> Clone for ArrayBase<S, D>
where
    S: Data + Clone,
    D: Dimension,
    S::Elem: Clone,
{
    fn clone(&self) -> Self {
        // 注意：对于视图，克隆会复制指针，导致两个视图共享同一数据（浅克隆）。
        // 对于自有数据，深克隆。
        Self {
            data: self.data.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            layout: self.layout,
        }
    }
}

impl<S, D> Index<&[usize]> for ArrayBase<S, D>
where
    S: Data,
    D: Dimension,
{
    type Output = S::Elem;
    fn index(&self, indices: &[usize]) -> &Self::Output {
        self.get(indices).expect("索引越界")
    }
}

impl<S, D> IndexMut<&[usize]> for ArrayBase<S, D>
where
    S: Data,
    D: Dimension,
{
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        self.get_mut(indices).expect("索引越界")
    }
}

// =============================================================================
// 加法运算 (完全模仿 ndarray，支持引用)
// =============================================================================

// 值 + 值
impl<T, D> Add for Array<T, D>
where
    T: Add<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;
    fn add(self, rhs: Self) -> Self::Output {
        self.add_ref(&rhs)
    }
}

// 值 + 引用
impl<T, D> Add<&Array<T, D>> for Array<T, D>
where
    T: Add<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;
    fn add(self, rhs: &Array<T, D>) -> Self::Output {
        self.add_ref(rhs)
    }
}

// 引用 + 值
impl<T, D> Add<Array<T, D>> for &Array<T, D>
where
    T: Add<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;
    fn add(self, rhs: Array<T, D>) -> Self::Output {
        rhs.add_ref(self)
    }
}

// 引用 + 引用
impl<T, D> Add<&Array<T, D>> for &Array<T, D>
where
    T: Add<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;
    fn add(self, rhs: &Array<T, D>) -> Self::Output {
        self.add_ref(rhs)
    }
}

// 具体实现（形状检查 + 逐元素加法）
impl<T, D> Array<T, D>
where
    T: Clone + Add<Output = T>,
    D: Dimension,
{
    // 注意：返回类型显式写为 Array<T, D>，避免歧义
    fn add_ref(&self, rhs: &Self) -> Array<T, D> {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "加法要求两个数组形状相同, 左: {:?}, 右: {:?}",
            self.shape(),
            rhs.shape()
        );
        let len = self.len();
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            let left = self.get_linear(i).expect("索引有效");
            let right = rhs.get_linear(i).expect("索引有效");
            data.push(left.clone() + right.clone());
        }
        Array::new(
            OwnedRepr::new(data),
            self.dim.clone(),
            self.strides.clone(),
            self.layout,
        )
    }
}

// =============================================================================
// reshape 功能 (into_shape)
// =============================================================================

#[derive(Debug, PartialEq, Eq)]
pub struct ShapeError;

impl<T, D> Array<T, D>
where
    D: Dimension,
{
    /// 重塑形状，不复制数据（要求内存连续且为行优先）。
    /// 仅适用于拥有所有权的数组 (`Array`)。
    pub fn into_shape<E: Dimension>(self, new_dim: E) -> Result<Array<T, E>, ShapeError> {
        let new_size = new_dim.size();
        if self.len() != new_size {
            return Err(ShapeError);
        }
        // 由于 `Array<T, D>` 使用的是 `OwnedRepr`，可以直接取回数据
        let OwnedRepr { data } = self.data;
        // 计算新的行优先步长
        let mut new_strides = vec![1; new_dim.ndim()];
        let mut stride = 1;
        for i in (0..new_dim.ndim()).rev() {
            new_strides[i] = stride;
            stride *= new_dim.shape()[i];
        }
        Ok(ArrayBase {
            data: OwnedRepr::new(data),
            dim: new_dim,
            strides: new_strides,
            layout: Layout::RowMajor,
        })
    }
}

// =============================================================================
// 便捷的类型别名
// =============================================================================

pub type Array<T, D> = ArrayBase<OwnedRepr<T>, D>;
pub type ArrayView<'a, T, D> = ArrayBase<ViewRepr<T>, D>;
pub type ArrayViewMut<'a, T, D> = ArrayBase<ViewRepr<T>, D>;

// =============================================================================
// 构造函数
// =============================================================================

impl<T> Array<T, Dim<1>> {
    pub fn from_vec(data: Vec<T>) -> Self {
        let dim = Dim::new([data.len()]);
        let strides = vec![1];
        Self::new(OwnedRepr::new(data), dim, strides, Layout::RowMajor)
    }
}

impl<T, const N: usize> Array<T, Dim<N>> {
    pub fn from_shape_vec(shape: [usize; N], data: Vec<T>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(data.len(), expected_len, "数据长度与形状不匹配");
        let dim = Dim::new(shape);
        let mut strides = vec![1; N];
        let mut stride = 1;
        for i in (0..N).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }
        Self::new(OwnedRepr::new(data), dim, strides, Layout::RowMajor)
    }

    pub fn from_elem(shape: [usize; N], elem: T) -> Self
    where
        T: Clone,
    {
        let len = shape.iter().product();
        let data = vec![elem; len];
        Self::from_shape_vec(shape, data)
    }
}

mod vecadd;
// =============================================================================
// 测试模块
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_array() {
        let arr = Array::from_vec(vec![10, 20, 30, 40]);
        assert_eq!(arr.shape(), &[4]);
        assert_eq!(arr[&[0]], 10);
        assert_eq!(arr[&[2]], 30);
    }

    #[test]
    fn test_2d_array() {
        let arr = Array::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[&[1, 2]], 6);
    }

    #[test]
    fn test_view() {
        let mut arr = Array::from_shape_vec([2, 2], vec![1, 2, 3, 4]);
        let view = arr.view();
        assert_eq!(view[&[1, 1]], 4);
        let mut view_mut = arr.view_mut();
        *view_mut.get_mut(&[1, 0]).unwrap() = 99;
        assert_eq!(arr[&[1, 0]], 99);
    }

    #[test]
    fn test_layoutref() {
        let arr = Array::from_shape_vec([2, 2], vec![1, 2, 3, 4]);
        let layout_ref = LayoutRef::new(arr.layout(), arr.strides());
        assert_eq!(layout_ref.layout(), Layout::RowMajor);
        assert_eq!(layout_ref.strides(), &[2, 1]);
    }

    #[test]
    fn test_add() {
        let a = Array::from_shape_vec([2, 2], vec![1, 2, 3, 4]);
        let b = Array::from_shape_vec([2, 2], vec![10, 20, 30, 40]);

        // 测试各种加法形式（现在 clone 可用）
        let c1 = &a + &b;
        let c2 = a.clone() + &b;
        let c3 = &a + b.clone();
        let c4 = a.clone() + b.clone();

        for c in [c1, c2, c3, c4] {
            assert_eq!(c[&[0, 0]], 11);
            assert_eq!(c[&[1, 1]], 44);
        }
    }

    #[test]
    #[should_panic(expected = "加法要求两个数组形状相同")]
    fn test_add_shape_mismatch() {
        let a = Array::from_shape_vec([2, 2], vec![1, 2, 3, 4]);
        let b = Array::from_vec(vec![1, 2, 3]); // 一维数组
        // 转换为动态维度使编译通过，运行时形状检查失败
        let a_dyn = a.into_shape(IxDyn::new(vec![2, 2])).unwrap();
        let b_dyn = b.into_shape(IxDyn::new(vec![3])).unwrap();
        let _ = &a_dyn + &b_dyn;
    }

    #[test]
    fn test_reshape() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
        let reshaped = arr.into_shape(Dim::new([2, 3])).unwrap();
        assert_eq!(reshaped.shape(), &[2, 3]);
        assert_eq!(reshaped[&[1, 2]], 6);

        let arr2 = Array::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]);
        let flat = arr2.into_shape(Dim::new([6])).unwrap();
        assert_eq!(flat[&[5]], 6);
    }

    #[test]
    fn test_reshape_error() {
        let arr = Array::from_vec(vec![1, 2, 3, 4]);
        let result = arr.into_shape(Dim::new([3, 2]));
        assert!(result.is_err());
    }

    #[test]
    fn test_from_elem() {
        let arr = Array::from_elem([2, 3], 5);
        assert_eq!(arr[&[1, 2]], 5);
    }
}