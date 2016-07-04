// extern crate serde_codegen;
//
// use std::env;
// use std::path::Path;
//
// pub fn main() {
//     let out_dir = env::var_os("OUT_DIR").unwrap();
//
//     let src = Path::new("src/main.rs.in");
//     let dst = Path::new(&out_dir).join("main.rs");
//
//     serde_codegen::expand(&src, &dst).unwrap();
// }

#[cfg(not(feature = "serde_macros"))]
mod inner {
    extern crate serde_codegen;

    use std::env;
    use std::path::Path;

    pub fn main() {
        let out_dir = env::var_os("OUT_DIR").unwrap();


        let libsrc = Path::new("src/clipper/lib.rs.in");
        let libdst = Path::new(&out_dir).join("lib.rs");

        serde_codegen::expand(&libsrc, &libdst).unwrap();

        let restbin_src = Path::new("src/bin/clipper-rest.rs.in");
        let restbin_dst = Path::new(&out_dir).join("clipper-rest.rs");

        serde_codegen::expand(&restbin_src, &restbin_dst).unwrap();
    }
}

#[cfg(feature = "serde_macros")]
mod inner {
    pub fn main() {}
}

fn main() {
    inner::main();
}
