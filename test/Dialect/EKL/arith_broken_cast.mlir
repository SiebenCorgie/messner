// RUN: messner-opt %s | FileCheck %s

ekl.program {
    //Broken for casting reasons
    ekl.func @brokenCast(%a: !ekl.array<f64[4, 4]>, %b: !ekl.array<f64[4, 4]>) -> i1{
        %0 = "ekl.min"(%a, %b) : (!ekl.array<f64[4, 4]>, !ekl.array<f64[4, 4]>) -> !ekl.array<f64[4, 4]>
        %1 = ekl.promote %a: !ekl.array<f64[4, 4]> -> i1
        yield %1: i1
    }
}

