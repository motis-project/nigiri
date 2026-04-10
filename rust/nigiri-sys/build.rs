use std::env;
use std::path::PathBuf;

fn main() {
    let nigiri_root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let header = nigiri_root.join("include/nigiri/abi.h");
    assert!(
        header.exists(),
        "Cannot find abi.h at {}",
        header.display()
    );

    // --- Generate bindings (always works — only needs the header) ---
    let bindings = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_type("nigiri_.*")
        .allowlist_function("nigiri_.*")
        .allowlist_var("kTargetBits|kDurationBits")
        .generate()
        .expect("Failed to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings");

    // --- Link against pre-built nigiri library ---
    // Set NIGIRI_BUILD_DIR to point at your CMake build output,
    // e.g. NIGIRI_BUILD_DIR=/workspaces/nigiri/build/relwithdebinfo
    println!("cargo:rerun-if-env-changed=NIGIRI_BUILD_DIR");

    let build_dir = env::var("NIGIRI_BUILD_DIR")
        .map(|p| {
            let pb = PathBuf::from(&p);
            if pb.is_relative() {
                nigiri_root.join(pb)
            } else {
                pb
            }
        })
        .ok()
        .or_else(|| {
            let default = nigiri_root.join("build/relwithdebinfo");
            if default.join("libnigiri.a").exists() {
                Some(default)
            } else {
                None
            }
        });

    if let Some(build_dir) = build_dir {
        link_prebuilt(&build_dir);
    } else {
        // No pre-built library found. Bindings are generated but linking
        // will be deferred. Build nigiri with CMake first, then set
        // NIGIRI_BUILD_DIR or place the build at build/relwithdebinfo.
        println!("cargo:warning=No pre-built nigiri library found. \
                  Build the C++ project first or set NIGIRI_BUILD_DIR.");
    }
}

fn link_prebuilt(build_dir: &PathBuf) {
    println!(
        "cargo:warning=Linking against pre-built nigiri in {}",
        build_dir.display()
    );

    // Collect all .a files in the build directory tree
    let mut lib_dirs = std::collections::HashSet::new();
    let mut lib_names = Vec::new();
    collect_static_libs(build_dir, &mut lib_dirs, &mut lib_names);

    // Emit search paths
    for dir in &lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir);
    }

    // Link nigiri first, then all dependency libs.
    // Use --whole-archive group to resolve circular deps between static libs.
    println!("cargo:rustc-link-lib=static=nigiri");
    for name in &lib_names {
        if *name != "nigiri" && *name != "gtest" && *name != "gmock" {
            println!("cargo:rustc-link-lib=static={}", name);
        }
    }

    // System libraries required by nigiri and its dependencies
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=z");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
}

fn collect_static_libs(
    dir: &PathBuf,
    lib_dirs: &mut std::collections::HashSet<String>,
    lib_names: &mut Vec<String>,
) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_static_libs(&path, lib_dirs, lib_names);
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if let Some(lib_name) =
                    name.strip_prefix("lib").and_then(|n| n.strip_suffix(".a"))
                {
                    lib_dirs.insert(
                        path.parent().unwrap().to_str().unwrap().to_string(),
                    );
                    lib_names.push(lib_name.to_string());
                }
            }
        }
    }
}
