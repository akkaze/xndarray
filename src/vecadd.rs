// use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
// use cuda_device::{kernel, thread, DisjointSlice};
// use cuda_host::{cuda_launch, load_kernel_module};

// /// Plain helper function -- no annotation needed.
// /// The compiler discovers it automatically because `vecadd` calls it.
// fn add(a: f32, b: f32) -> f32 {
//     a + b
// }

// #[kernel]
// pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
//     let idx = thread::index_1d();
//     if let Some(c_elem) = c.get_mut(idx) {
//         *c_elem = add(a[idx.get()], b[idx.get()]);
//     }
// }

// #[cfg(test)]
// mod test {
//     use super::*;
//     #[test]
//     fn test_vecadd_kernel() -> Result<(), Box<dyn std::error::Error>> {
//         let ctx = CudaContext::new(0).unwrap();
//         let stream = ctx.default_stream();

//         const N: usize = 1024;
//         let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
//         let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

//         let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
//         let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
//         let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

//         // Loads `my_first_kernel.ptx` directly when cuda-oxide produced PTX, or
//         // builds a cubin from `my_first_kernel.ll` when cuda-oxide auto-detected
//         // CUDA libdevice math (`sin`, `pow`, `exp`, ...). Either way one call.
//         let module =
//             load_kernel_module(&ctx, "xndarray").expect("Failed to load kernel module");

//         cuda_launch! {
//             kernel: vecadd,
//             stream: stream,
//             module: module,
//             config: LaunchConfig::for_num_elems(N as u32),
//             args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
//         }
//         .unwrap();

//         let c_host = c_dev.to_host_vec(&stream).unwrap();
//         let errors = (0..N)
//             .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
//             .count();

//         assert_eq!(errors, 0, "向量加法内核输出不正确");
//         Ok(())
//     }
// }
