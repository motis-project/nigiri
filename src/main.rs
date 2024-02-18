#[cxx::bridge]
mod nigiri {
    unsafe extern "C++" {
        include!("nigiri/rust.h");

        type Timetable;

        fn new_timetable(paths: &Vec<String>) -> UniquePtr<Timetable>;
    }
}

fn main() {
    let _tt = nigiri::new_timetable(&vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    println!("Hello, world!");
}
