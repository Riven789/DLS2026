use matrixmultiply::dgemm;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2,fma")]
#[cfg(target_arch = "x86_64")]
unsafe fn gemm_simd(a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    for i in 0..n {
        for k in 0..n {
            let aik = a[i*n+k];
            let aik_vec = _mm256_set1_pd(aik);
            let mut j = 0;
            while j + 4 <= n {
                let b_vec = _mm256_loadu_pd(b.as_ptr().add(k*n+j));
                let c_vec = _mm256_loadu_pd(c.as_ptr().add(i*n+j));
                let res = _mm256_fmadd_pd(aik_vec, b_vec, c_vec);
                _mm256_storeu_pd(c.as_mut_ptr().add(i*n+j), res);
                j += 4;
            }
            while j < n {
                c[i*n+j] += aik * b[k*n+j];
                j += 1;
            }
        }
    }
}

fn align_64(mut v: Vec<f64>, n: usize) -> Vec<f64> {
    let off = (64 - (v.as_ptr() as usize % 64)) % 64 / 8;
    v.drain(0..off);
    v.truncate(n);
    v
}

fn main() {
    let n = 1024usize;
    let a = align_64(vec![1.0f64; n * n + 64], n * n);
    let b = align_64(vec![2.0f64; n * n + 64], n * n);

    // 1) BLAS multi-thread (env-controlled)
    let mut c1 = vec![0.0f64; n * n];
    std::env::set_var("OPENBLAS_NUM_THREADS", "10");
    let t1 = std::time::Instant::now();
    unsafe {
        dgemm(n, n, n, 1.0, a.as_ptr(), n as isize, 1,
              b.as_ptr(), n as isize, 1, 1.0,
              c1.as_mut_ptr(), n as isize, 1);
    }
    println!("BLAS multi-threaded (env): {:?} c[0]={}", t1.elapsed(), c1[0]);

    // 2) Rayon row-block split (each thread does sub-GEMM on rows)
    let mut c2 = vec![0.0f64; n * n];
    let tile = 256usize;
    let t2 = std::time::Instant::now();
    let c2_raw: usize = c2.as_mut_ptr() as usize;
    (0..n).into_par_iter().step_by(tile).for_each(|ii| {
        let i_max = (ii + tile).min(n);
        for jj in (0..n).step_by(tile) {
            let j_max = (jj + tile).min(n);
            let mut sub = vec![0.0f64; (i_max - ii) * (j_max - jj)];
            unsafe {
                dgemm(i_max - ii, n, j_max - jj, 1.0,
                      a.as_ptr().add(ii * n), n as isize, 1,
                      b.as_ptr().add(jj), n as isize, 1,
                      0.0, sub.as_mut_ptr(), (j_max - jj) as isize, 1);
                for r in 0..(i_max - ii) {
                    let row_base = (ii + r) * n + jj;
                    std::ptr::copy_nonoverlapping(sub.as_ptr().add(r * (j_max - jj)), (c2_raw as *mut f64).add(row_base), j_max - jj);
                }
            }
        }
    });
    println!("Rayon row-block: {:?} (partial)", t2.elapsed());

    // 5) AVX2 SIMD
    let mut c3 = vec![0.0f64; n*n];
    let t3 = std::time::Instant::now();
    unsafe { gemm_simd(&a, &b, &mut c3, n); }
    println!("AVX2 SIMD: {:?} c[0]={}", t3.elapsed(), c3[0]);
}
