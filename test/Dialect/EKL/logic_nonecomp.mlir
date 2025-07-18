// RUN: messner-opt %s | FileCheck %s

ekl.program {
    
    ekl.func @myfunc(%a: !ekl.array<f64[4, 4]>, %b: !ekl.array<ui64[4, 4]>) -> !ekl.array<i1[4, 4]>{
        %0 = "ekl.cmp"(%a, %b) {kind = 0} : (!ekl.array<f64[4, 4]>, !ekl.array<ui64[4, 4]>) -> !ekl.array<i1[4, 4]> 
        yield %0: !ekl.array<i1[4, 4]>
    }
}

