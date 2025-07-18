// RUN: messner-opt %s | FileCheck %s

ekl.program {
    //Triggers the bool special case when typchecking the cmp.kind.
    ekl.func @myfunc(%a: !ekl.array<i1[4, 4]>, %b: !ekl.array<i1[4, 4]>) -> !ekl.array<i1[4, 4]>{
        %0 = "ekl.cmp"(%a, %b) {kind = 2} : (!ekl.array<i1[4, 4]>, !ekl.array<i1[4, 4]>) -> !ekl.array<i1[4, 4]> 
        yield %0: !ekl.array<i1[4, 4]>
    }
}

