/*
 * This file is distributed under the Apache License, Version 2.0; refer to
 * LICENSE for details.
 *
 * Initial author: Emanuel Vintila
 */

#include "use_after_star.h"

#include <cassert>
#include <string>
#include <vector>

#include "misc.h"
#include "generator/primitives/bug_types/temporal/memory_state/freed_memory.h"
#include "generator/primitives/bug_types/temporal/memory_state/used.h"
#include "generator/primitives/regions/heap_region.h"
#include "generator/primitives/regions/stack_region.h"

const std::string max_reallocated_retries = "1000000000";
const std::string max_reallocated_retries_validation = "100";

UseAfterStar::UseAfterStar():
  TemporalBugType("use_after_star")
{
}

bool UseAfterStar::accepts(std::shared_ptr<MemoryState> memory_state)
{
  return true;
}

bool UseAfterStar::accepts(std::shared_ptr<Region> region)
{
  return is_a<HeapRegion>(region)
    || is_a<StackRegion>(region);
}

bool UseAfterStar::accepts(std::shared_ptr<AccessLocation> access_location)
{
  return true;
}

std::vector< std::shared_ptr<RegionCodeCanvas> >UseAfterStar::generate(
  std::shared_ptr<MemoryState> memory_state,
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location
  ) const
{
  if (is_a<FreedMemory>(memory_state))
  {
    return _generate_unused_mem(memory_region, access_action, access_location);
  }
  assert(is_a<UsedMemory>(memory_state));
  return _generate_reused_mem(memory_region, access_action, access_location);
}

std::vector< std::shared_ptr<RegionCodeCanvas> >UseAfterStar::_generate_unused_mem(
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location) const
{
  if ( is_a<HeapRegion>(memory_region) ) return _generate_unused_mem_heap(memory_region, access_action, access_location);
  assert( is_a<StackRegion>(memory_region) );
  return _generate_unused_mem_stack(memory_region, access_action, access_location);
}

std::vector< std::shared_ptr<RegionCodeCanvas> >UseAfterStar::_generate_unused_mem_stack(
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location) const
{

  std::vector< std::shared_ptr<RegionCodeCanvas> >variants = {};
  std::shared_ptr<StackRegion> stack_memory_region = std::dynamic_pointer_cast<StackRegion>(memory_region);
  std::shared_ptr<StackRegion> stack_memory_region_simple = std::make_shared<StackRegion>(*stack_memory_region);
  std::shared_ptr<RegionCodeCanvas> region_canvas = stack_memory_region_simple->generate(std::make_shared<CodeCanvas>(CodeCanvas()), "target", 8, true);

  // region_canvas->add_during_lifetime({
  //   "%c0_v = arith.constant 0 : index",
  //   "%c8_v = arith.constant 8 : index",
  //   "%assert_in_use_after_scope = arith.cmpi sge,%c8_v, %c0_v : index",
  //   "scf.if %assert_in_use_after_scope {",
  //   "  func.return %target : memref<8xi8>",
  //   "} else {",
  //   "  func.call @exit(%precond_fail) : (i32) -> ()",
  //   "}"
  // });
  
  AccessLocation::SplitAccess access_type_code = access_location->generate_split_const_vars(
    access_action, "ret", 8, 0, "", false, "0", "memref<8xi8>");

  access_type_code.access_lines.insert(access_type_code.access_lines.begin(), "%test_success = arith.constant 42 : i32");
  CodeCanvas::code_pos_t index = region_canvas->add_at(region_canvas->get_f_call_pos() + 1, access_type_code.to_lines(), "    ");
  region_canvas->add_at(index, "func.call @exit(%test_success) : (i32) -> ()", "    ");

  region_canvas->add_test_case_description_line("Memory region: stack");
  region_canvas->add_test_case_description_line("Bug type: use-after-*, freed memory");
  region_canvas->add_test_case_description_line("Access type: " + access_location->get_name() + ", " + access_action->get_name());

  variants.push_back( region_canvas );

  return variants;
}


std::vector< std::shared_ptr<RegionCodeCanvas> >UseAfterStar::_generate_unused_mem_heap(
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location) const
{
  std::vector< std::shared_ptr<RegionCodeCanvas> >variants;
  CodeCanvas code;
  std::shared_ptr<RegionCodeCanvas> region_canvas = memory_region->generate(std::make_shared<CodeCanvas>(code), "target", 8, true);

  std::vector<std::string> post_dealloc = {
    "%c0_v = arith.constant 0 : index",
    "%c8_v = arith.constant 8 : index",
    "%view_target = memref.view %target[%c0_v][%c8_v] : memref<8xi8> to memref<?xi8>"
  };
  auto index = region_canvas->add_at(region_canvas->get_deallocation_pos() + 1, post_dealloc, "    ");

  std::vector<std::string> access_type_code = access_location->generate(
    access_action, "view_target", 8, 0, "", false, "0", "memref<?xi8>");
  index = region_canvas->add_at(index, access_type_code, "    ");
  region_canvas->add_at(index, "func.call @exit(%test_success) : (i32) -> ()", "    ");

  region_canvas->add_test_case_description_line("Memory region: heap");
  region_canvas->add_test_case_description_line("Bug type: use-after-*, freed memory");
  region_canvas->add_test_case_description_line("Access type: " + access_location->get_name() + ", " + access_action->get_name());

  variants.push_back( region_canvas );

  return variants;
}


std::vector< std::shared_ptr<RegionCodeCanvas> >UseAfterStar::_generate_reused_mem(
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location) const
{
  if ( is_a<HeapRegion>(memory_region) ) return _generate_reused_mem_heap(memory_region, access_action, access_location);
  assert( is_a<StackRegion>(memory_region) );
  return _generate_reused_mem_stack(memory_region, access_action, access_location);
}

std::vector< std::shared_ptr<RegionCodeCanvas> >UseAfterStar::_generate_reused_mem_heap(
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location) const
{
  std::vector< std::shared_ptr<RegionCodeCanvas> >variants;
  CodeCanvas code;
  std::shared_ptr<RegionCodeCanvas> region_canvas = memory_region->generate(std::make_shared<CodeCanvas>(code), "target", 8, false);

  assert ( is_a<HeapRegion>(memory_region) );

  region_canvas->add_locals({
    "%c0_v = arith.constant 0 : index",
    "%c8_v = arith.constant 8 : index",
    "%view_target = memref.view %target[%c0_v][%c8_v] : memref<8xi8> to memref<?xi8>"
  });

  std::vector<std::string> access_type_code = access_location->generate(
    access_action, "view_target", 8, 0, "", false, "0", "memref<?xi8>");
  std::shared_ptr<HeapRegion> heap_memory_region = std::dynamic_pointer_cast<HeapRegion>(memory_region);

  std::shared_ptr<RegionCodeCanvas> reused_region_canvas = heap_memory_region->generate(
    region_canvas->get_deallocation_pos(), region_canvas, "reallocated", 8, false);
  std::shared_ptr<RegionCodeCanvas> reused_region_canvas_simple = std::make_shared<RegionCodeCanvas>(*reused_region_canvas);

  reused_region_canvas_simple->add_during_lifetime({
    "%target_ptr = memref.extract_aligned_pointer_as_index %target : memref<8xi8> -> index",
    "%realloc_ptr = memref.extract_aligned_pointer_as_index %reallocated : memref<8xi8> -> index",
    "%eq = arith.cmpi eq, %target_ptr, %realloc_ptr : index",
    "%ctrue = arith.constant 1 : i1",
    "%neq = arith.xori %eq, %ctrue : i1",
    "scf.if %neq {",
    "  func.call @exit(%precond_fail) : (i32) -> ()",
    "}"
  });

  reused_region_canvas_simple->add_during_lifetime(access_type_code);
  reused_region_canvas_simple->add_during_lifetime("func.call @exit(%test_success) : (i32) -> ()");

  region_canvas->add_test_case_description_line("Memory region: heap");
  region_canvas->add_test_case_description_line("Bug type: use-after-*, reused memory");
  region_canvas->add_test_case_description_line("Access type: " + access_location->get_name() + ", " + access_action->get_name());

  std::shared_ptr<RegionCodeCanvas> reused_region_canvas_repeat = std::make_shared<RegionCodeCanvas>(*reused_region_canvas);
  std::vector<std::string> allocation = heap_memory_region->generate_reallocation("new_reallocated", 8, true, "    ");

  reused_region_canvas_repeat->add_during_lifetime({
    "%c0 = arith.constant 0 : index",
    "%c1 = arith.constant 1 : index",
    "%c_max = arith.constant " + max_reallocated_retries + " : index",
    "%ctrue_loop = arith.constant 1 : i1"
  });

  std::vector<std::string> allocation_for = heap_memory_region->generate_reallocation("new_reallocated", 8, true, "    ");
  reused_region_canvas_repeat->add_during_lifetime({ //TODO: 这里替换为scf.for的写法
    "%reallocated_for, %not_matched_for = scf.for %counter = %c0 to %c_max step %c1",
    "    iter_args(%reallocated_iter = %reallocated, %not_matched_iter = %ctrue_loop)",
    "    -> (memref<8xi8>, i1) {",
    "  %next_reallocated, %next_not_matched = scf.if %not_matched_iter -> (memref<8xi8>, i1) {",
    "    memref.dealloc %reallocated_iter : memref<8xi8>"
  });

  reused_region_canvas_repeat->add_during_lifetime(allocation_for);

  reused_region_canvas_repeat->add_during_lifetime({
    "    %target_ptr_loop = memref.extract_aligned_pointer_as_index %target : memref<8xi8> -> index",
    "    %new_ptr_loop = memref.extract_aligned_pointer_as_index %new_reallocated : memref<8xi8> -> index",
    "    %eq_loop = arith.cmpi eq, %target_ptr_loop, %new_ptr_loop : index",
    "    %not_matched_next = arith.xori %eq_loop, %ctrue_loop : i1",
    "    scf.yield %new_reallocated, %not_matched_next : memref<8xi8>, i1",
    "  } else {",
    "    scf.yield %reallocated_iter, %not_matched_iter : memref<8xi8>, i1",
    "  }",
    "  scf.yield %next_reallocated, %next_not_matched : memref<8xi8>, i1",
    "}"
  });

  reused_region_canvas_repeat->add_during_lifetime({
    "scf.if %not_matched_for {",
    "  func.call @exit(%precond_fail) : (i32) -> ()",
    "}"
  });

  reused_region_canvas_repeat->add_during_lifetime(access_type_code);
  reused_region_canvas_repeat->add_during_lifetime("func.call @exit(%test_success) : (i32) -> ()");

  reused_region_canvas_repeat->add_variant_description_line("with repeated attempts");

  variants = {reused_region_canvas_simple, reused_region_canvas_repeat};

  return variants;
}

std::vector< std::shared_ptr<RegionCodeCanvas>>UseAfterStar::_generate_reused_mem_stack(
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location) const
{
  // Stack reused variants are simplified for now (kept minimal to compile)
  std::vector< std::shared_ptr<RegionCodeCanvas> >variants = {};
  CodeCanvas code_simple;
  std::shared_ptr<StackRegion> stack_memory_region = std::dynamic_pointer_cast<StackRegion>(memory_region);
  std::shared_ptr<StackRegion> stack_memory_region_simple = std::make_shared<StackRegion>(*stack_memory_region);
  std::shared_ptr<RegionCodeCanvas> region_canvas = stack_memory_region_simple->generate(std::make_shared<CodeCanvas>(code_simple), "target", 8, true);
  region_canvas->add_global("memref.global @target_addr : memref<8xi8> = dense<170>");
  region_canvas->add_during_lifetime({
    "%test_success = arith.constant 42 : i32",
"%c0_v = arith.constant 0 : index",
    "%c8_v = arith.constant 8 : index",
    "%view_target = memref.view %target[%c0_v][%c8_v] : memref<8xi8> to memref<?xi8>"
  });
  AccessLocation::SplitAccess access_type_code = access_location->generate_split_const_vars(
    access_action, "target_addr", 8, 0, "", false, "0", "memref<8xi8>");

  std::shared_ptr<RegionCodeCanvas> reused_region_canvas_simple = stack_memory_region_simple->generate_in_other_f(
    region_canvas, "reallocated", 8, false
  );
  reused_region_canvas_simple->add_during_lifetime({
    "%precond_fail = arith.constant 43 : i32",
    "%target_addr = memref.get_global @target_addr : memref<8xi8>",
    "%target_ptr = memref.extract_aligned_pointer_as_index %target_addr : memref<8xi8> -> index",
    "%realloc_ptr = memref.extract_aligned_pointer_as_index %reallocated : memref<8xi8> -> index",
    "%eq = arith.cmpi eq, %target_ptr, %realloc_ptr : index",
    "%ctrue = arith.constant 1 : i1",
    "%neq = arith.xori %eq, %ctrue : i1",
    "scf.if %neq {",
    "  func.call @exit(%precond_fail) : (i32) -> ()",
    "}"
  });
  access_type_code.access_lines.insert(access_type_code.access_lines.begin(), "%test_success = arith.constant 42 : i32");
  reused_region_canvas_simple->add_during_lifetime(access_type_code.to_lines());
  reused_region_canvas_simple->add_during_lifetime("func.call @exit(%test_success) : (i32) -> ()");

  region_canvas->add_test_case_description_line("Memory region: stack");
  region_canvas->add_test_case_description_line("Bug type: use-after-*, reused memory");
  region_canvas->add_test_case_description_line("Access type: " + access_location->get_name() + ", " + access_action->get_name());

  variants.push_back( reused_region_canvas_simple );

  std::shared_ptr<RegionCodeCanvas> reused_region_canvas_repeated = stack_memory_region_simple->generate_in_other_f(
    region_canvas, "reallocated", 8, false
    );
  reused_region_canvas_repeated->add_global("memref.global @last_address : memref<8xi8> = dense<0>");
  reused_region_canvas_repeated->add_during_lifetime({
            "%test_success = arith.constant 42 : i32",
"%precond_fail = arith.constant 43 : i32",
    "%target_addr = memref.get_global @target_addr : memref<8xi8>",
    "%target_ptr = memref.extract_aligned_pointer_as_index %target_addr : memref<8xi8> -> index",
    "%realloc_ptr = memref.extract_aligned_pointer_as_index %reallocated : memref<8xi8> -> index",
    "%last_address = memref.get_global @last_address : memref<8xi8>",
    "%last_ptr = memref.extract_aligned_pointer_as_index %last_address : memref<8xi8> -> index",
    "%eq = arith.cmpi eq, %target_ptr, %realloc_ptr : index",
    "%eq_1= arith.cmpi eq, %target_ptr, %last_ptr : index",
    "%ctrue = arith.constant 1 : i1",
    "%neq = arith.xori %eq, %ctrue : i1",
    "%neq_1 = arith.xori %eq_1, %ctrue : i1",
    "scf.if %neq_1 {",
    "  func.call @exit(%precond_fail) : (i32) -> ()",
    "}",
    "%c0 = arith.constant 0 : index",
    "%val_realloc = memref.load %reallocated[%c0] : memref<8xi8>",
    "memref.store %val_realloc, %last_address[%c0] : memref<8xi8>",
    "scf.if %neq {",
    "  func.call @exit(%precond_fail) : (i32) -> ()",
    "}",
    
  });
  reused_region_canvas_repeated->add_during_lifetime(access_type_code.to_lines());
  reused_region_canvas_repeated->add_during_lifetime("func.call @exit(%test_success) : (i32) -> ()");
  reused_region_canvas_repeated->add_at(reused_region_canvas_repeated->get_other_f_call_pos(),
    std::vector<std::string>{//TODO: 这里替换为scf.for的写法
    "%precond_fail = arith.constant 43 : i32",
      "%c0 = arith.constant 0 : index",
      "%c1 = arith.constant 1 : index",
      "%c_max = arith.constant " + max_reallocated_retries + " : index",
      "%ctrue = arith.constant 1 : i1",
      "%not_matched_for = scf.for %counter = %c0 to %c_max step %c1",
      "    iter_args(%not_matched_iter = %ctrue)",
      "    -> (i1) {",
      "  %next_not_matched = scf.if %not_matched_iter -> (i1) {"
    },
    "  "
  );
  reused_region_canvas_repeated->add_at(reused_region_canvas_repeated->get_other_f_call_pos() + 1,
    std::vector<std::string>{
      "    %cfalse = arith.constant 0 : i1",
      "    scf.yield %cfalse : i1",
      "  } else {",
      "    scf.yield %not_matched_iter : i1",
      "  }",
      "  scf.yield %next_not_matched : i1",
      "}",
      "scf.if %not_matched_for {",
      "  func.call @exit(%precond_fail) : (i32) -> ()",
      "}"
    },
    "  "
  );
  reused_region_canvas_repeated->add_variant_description_line("with repeated attempts");
  variants.push_back( reused_region_canvas_repeated );

  CodeCanvas code_array;
  code_array.add_global("memref.global @last_address : memref<8xi8> = dense<0>");
  code_array.add_global("memref.global @target_addresses : memref<16x8xi8> = dense<0>");
  std::shared_ptr<StackRegion> stack_memory_region_array = std::make_shared<StackRegion>(*stack_memory_region);
  std::shared_ptr<RegionCodeCanvas> array_region_canvas = stack_memory_region_array->generate_array(std::make_shared<CodeCanvas>(code_array), "target", 8, 16, true);

  array_region_canvas->add_during_lifetime({
    "%c0_v = arith.constant 0 : index",
    "%c8_v = arith.constant 8 : index",
    "%view_target = memref.subview %target[0,0][16,8][1,1] : memref<16x8xi8> to memref<16x8xi8>"
  });
  AccessLocation::SplitAccess access_type_code_array = access_location->generate_split_const_vars(
    access_action, "view_target", 8, 16, "c0", false, "0", "memref<16x8xi8>");
  std::shared_ptr<RegionCodeCanvas> reused_region_canvas_simple_array = stack_memory_region_array->generate_in_other_f(
    array_region_canvas, "reallocated", 8, false
  );

  reused_region_canvas_simple_array->add_during_lifetime({
    "%test_success = arith.constant 42 : i32",
    "%precond_fail = arith.constant 43 : i32",
    "%c0 = arith.constant 0 : index",
    "%c1 = arith.constant 1 : index",
    "%c8 = arith.constant 8 : index",
    "%c16 = arith.constant 16 : index",
    "%cfalse = arith.constant 0 : i1",
    "%view_target = memref.get_global @target_addresses : memref<16x8xi8>",
    "%found = scf.for %counter = %c0 to %c16 step %c1 iter_args(%found_iter = %cfalse) -> (i1) {",
    "  %realloc_ptr = memref.extract_aligned_pointer_as_index %reallocated : memref<8xi8> -> index",
    "  %target_base = memref.extract_aligned_pointer_as_index %view_target : memref<16x8xi8> -> index",
    "  %offset = arith.muli %counter, %c8 : index",
    "  %target_ptr = arith.addi %target_base, %offset : index",
    "  %eq = arith.cmpi eq, %realloc_ptr, %target_ptr : index",
    "  %found_next = arith.ori %found_iter, %eq : i1",
    "  scf.yield %found_next : i1",
    "}",
    "scf.if %found {",
    "} else {",
    "  func.call @exit(%precond_fail) : (i32) -> ()",
    "}"
  });
  reused_region_canvas_simple_array->add_during_lifetime(access_type_code_array.to_lines());
  reused_region_canvas_simple_array->add_during_lifetime("func.call @exit(%test_success) : (i32) -> ()");
  reused_region_canvas_simple_array->add_variant_description_line("using an array of objects");
  variants.push_back( reused_region_canvas_simple_array );


  std::shared_ptr<StackRegion> stack_memory_region_array_repeated = std::make_shared<StackRegion>(*stack_memory_region);
  array_region_canvas = stack_memory_region_array_repeated->generate_array(std::make_shared<CodeCanvas>(code_array), "target", 8, 16, true);
  array_region_canvas->add_during_lifetime({
      "%c0_v = arith.constant 0 : index",
      "%c8_v = arith.constant 8 : index",
      "%view_target = memref.subview %target[0,0][16,8][1,1] : memref<16x8xi8> to memref<16x8xi8>"
  });  
   AccessLocation::SplitAccess access_type_code_array_repeated = access_location->generate_split_const_vars(
    access_action, "view_target", 8, 16, "c0", false, "0", "memref<16x8xi8>");
  std::shared_ptr<RegionCodeCanvas> reused_region_canvas_array_repeated = stack_memory_region_array->generate_in_other_f(
    array_region_canvas, "reallocated", 8, false
  );
  reused_region_canvas_array_repeated->add_during_lifetime({
            "%test_success = arith.constant 42 : i32",
"%precond_fail = arith.constant 43 : i32",
    "%last_address = memref.get_global @last_address : memref<8xi8>",
    "%last_ptr = memref.extract_aligned_pointer_as_index %last_address : memref<8xi8> -> index",
    "%realloc_ptr_check = memref.extract_aligned_pointer_as_index %reallocated : memref<8xi8> -> index",
    "%eq_last = arith.cmpi eq, %last_ptr, %realloc_ptr_check : index",
    "scf.if %eq_last {",
    "  func.call @exit(%precond_fail) : (i32) -> ()",
    "}",
    "%c0 = arith.constant 0 : index",
    "%c1 = arith.constant 1 : index",
    "%c8 = arith.constant 8 : index",
    "%c16 = arith.constant 16 : index",
    "%cfalse = arith.constant 0 : i1",
    "%view_target = memref.get_global @target_addresses : memref<16x8xi8>",
    "%found = scf.for %counter = %c0 to %c16 step %c1 iter_args(%found_iter = %cfalse) -> (i1) {",
    "  %realloc_ptr = memref.extract_aligned_pointer_as_index %reallocated : memref<8xi8> -> index",
    "  %target_base = memref.extract_aligned_pointer_as_index %view_target : memref<16x8xi8> -> index",
    "  %offset = arith.muli %counter, %c8 : index",
    "  %target_ptr = arith.addi %target_base, %offset : index",
    "  %eq = arith.cmpi eq, %realloc_ptr, %target_ptr : index",
    "  %found_next = arith.ori %found_iter, %eq : i1",
    "  scf.yield %found_next : i1",
    "}",
    "scf.if %found {",
    "} else {",
    "  func.call @exit(%precond_fail) : (i32) -> ()",
    "}"
  });
  reused_region_canvas_array_repeated->add_during_lifetime("func.call @exit(%test_success) : (i32) -> ()");
  reused_region_canvas_array_repeated->add_at(reused_region_canvas_array_repeated->get_other_f_call_pos(),
    std::vector<std::string>{//TODO: 这里替换为scf.for的写法
    "%precond_fail = arith.constant 43 : i32",
      "%c0 = arith.constant 0 : index",
      "%c1 = arith.constant 1 : index",
      "%c_max = arith.constant " + max_reallocated_retries + " : index",
      "%ctrue = arith.constant 1 : i1",
      "%not_matched_for = scf.for %counter = %c0 to %c_max step %c1",
      "    iter_args(%not_matched_iter = %ctrue)",
      "    -> (i1) {",
      "  %next_not_matched = scf.if %not_matched_iter -> (i1) {"
    },
    "  "
  );
  reused_region_canvas_array_repeated->add_at(reused_region_canvas_array_repeated->get_other_f_call_pos() + 1,
    std::vector<std::string>{
      "    %cfalse = arith.constant 0 : i1",
      "    scf.yield %cfalse : i1",
      "  } else {",
      "    scf.yield %not_matched_iter : i1",
      "  }",
      "  scf.yield %next_not_matched : i1",
      "}",
      "scf.if %not_matched_for {",
      "  func.call @exit(%precond_fail) : (i32) -> ()",
      "}"
    },
    "  "
  );
  reused_region_canvas_array_repeated->add_during_lifetime(access_type_code_array_repeated.to_lines());

  reused_region_canvas_array_repeated->add_variant_description_line("with repeated attempts");
  reused_region_canvas_array_repeated->add_variant_description_line("using an array of objects");

  variants.push_back( reused_region_canvas_array_repeated );

  return variants;
}


std::vector< std::shared_ptr<RegionCodeCanvas> >UseAfterStar::generate_validation(
  std::shared_ptr<MemoryState> memory_state,
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location
  ) const
{
  if (is_a<FreedMemory>(memory_state))
  {
    return _generate_unused_mem_validation(memory_region, access_action, access_location);
  }
  assert(is_a<UsedMemory>(memory_state));
  return _generate_reused_mem_validation(memory_region, access_action, access_location);
}


std::vector< std::shared_ptr<RegionCodeCanvas> >UseAfterStar::_generate_unused_mem_validation(
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location) const
{
  if ( is_a<HeapRegion>(memory_region) ) return _generate_unused_mem_heap_validation(memory_region, access_action, access_location);
  assert( is_a<StackRegion>(memory_region) );
  return _generate_unused_mem_stack_validation(memory_region, access_action, access_location);
}

std::vector< std::shared_ptr<RegionCodeCanvas> >UseAfterStar::_generate_unused_mem_stack_validation(
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location) const
{

  std::vector< std::shared_ptr<RegionCodeCanvas> >variants = {};
  CodeCanvas code_simple;
  std::shared_ptr<StackRegion> stack_memory_region = std::dynamic_pointer_cast<StackRegion>(memory_region);
  std::shared_ptr<StackRegion> stack_memory_region_simple = std::make_shared<StackRegion>(*stack_memory_region);
  std::shared_ptr<RegionCodeCanvas> region_canvas = stack_memory_region_simple->generate(std::make_shared<CodeCanvas>(code_simple), "target", 8, true);

  region_canvas->add_to_f_body({
    "%c0_v = arith.constant 0 : index",
    "%c8_v = arith.constant 8 : index",
    "%view_target = memref.view %target[%c0_v][%c8_v] : memref<8xi8> to memref<?xi8>"
  });
  AccessLocation::SplitAccess access_type_code = access_location->generate_split_const_vars(
    access_action, "view_target", 8, 0, "", false, "0", "memref<?xi8>");

  region_canvas->add_to_f_body(access_type_code.to_lines());
  region_canvas->add_to_f_body("func.call @exit(%test_success) : (i32) -> ()");

  region_canvas->add_test_case_description_line("Memory region: stack");
  region_canvas->add_test_case_description_line("Bug type: use-after-*, freed memory");
  region_canvas->add_test_case_description_line("Access type: " + access_location->get_name() + ", " + access_action->get_name());


  variants.push_back( region_canvas );


  return variants;
}


std::vector< std::shared_ptr<RegionCodeCanvas> >UseAfterStar::_generate_unused_mem_heap_validation(
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location) const
{
  std::vector< std::shared_ptr<RegionCodeCanvas> >variants;
  CodeCanvas code;
  std::shared_ptr<RegionCodeCanvas> region_canvas = memory_region->generate(std::make_shared<CodeCanvas>(code), "target", 8, /*initialize=*/true);

  assert( is_a<HeapRegion>(memory_region) );

  region_canvas->add_to_f_body({
    "%c0_v = arith.constant 0 : index",
    "%c8_v = arith.constant 8 : index",
    "%view_target = memref.view %target[%c0_v][%c8_v] : memref<8xi8> to memref<?xi8>"
  });

  std::vector<std::string> access_type_code = access_location->generate(
    access_action, "view_target", 8, 0, "", false, "0", "memref<?xi8>");
  auto index = region_canvas->add_to_f_body(access_type_code);
  region_canvas->add_at(index, "func.call @exit(%test_success) : (i32) -> ()", "    ");

  region_canvas->add_test_case_description_line("Memory region: heap");
  region_canvas->add_test_case_description_line("Bug type: use-after-*, freed memory");
  region_canvas->add_test_case_description_line("Access type: " + access_location->get_name() + ", " + access_action->get_name());

  variants.push_back( region_canvas );

  return variants;
}


std::vector< std::shared_ptr<RegionCodeCanvas> >UseAfterStar::_generate_reused_mem_validation(
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location) const
{
  if ( is_a<HeapRegion>(memory_region) ) return _generate_reused_mem_heap_validation(memory_region, access_action, access_location);
  assert( is_a<StackRegion>(memory_region) );
  return _generate_reused_mem_stack_validation(memory_region, access_action, access_location);
}

std::vector< std::shared_ptr<RegionCodeCanvas> >UseAfterStar::_generate_reused_mem_heap_validation(
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location) const
{
  std::vector< std::shared_ptr<RegionCodeCanvas> >variants;
  CodeCanvas code;
  std::shared_ptr<RegionCodeCanvas> region_canvas = memory_region->generate(std::make_shared<CodeCanvas>(code), "target", 8, false);

  assert ( is_a<HeapRegion>(memory_region) );

  region_canvas->add_locals({
    "%c0_v = arith.constant 0 : index",
    "%c8_v = arith.constant 8 : index",
    "%view_target = memref.view %target[%c0_v][%c8_v] : memref<8xi8> to memref<?xi8>"
  });

  std::vector<std::string> access_type_code = access_location->generate(
    access_action, "view_target", 8, 0, "", false, "0", "memref<?xi8>");
  std::shared_ptr<HeapRegion> heap_memory_region = std::dynamic_pointer_cast<HeapRegion>(memory_region);

  std::shared_ptr<RegionCodeCanvas> reused_region_canvases = heap_memory_region->generate(
    region_canvas->get_lifetime_pos(), region_canvas, "reallocated", 8, false);
  std::shared_ptr<RegionCodeCanvas> reused_region_canvas_simple = std::make_shared<RegionCodeCanvas>(*reused_region_canvases);

  reused_region_canvas_simple->add_during_lifetime(access_type_code);
  reused_region_canvas_simple->add_during_lifetime("func.call @exit(%test_success) : (i32) -> ()");

  region_canvas->add_test_case_description_line("Memory region: heap");
  region_canvas->add_test_case_description_line("Bug type: use-after-*, reused memory");
  region_canvas->add_test_case_description_line("Access type: " + access_location->get_name() + ", " + access_action->get_name());


  std::shared_ptr<RegionCodeCanvas> reused_region_canvas_repeat = std::make_shared<RegionCodeCanvas>(*reused_region_canvases);
  std::vector<std::string> allocation = heap_memory_region->generate_reallocation("new_reallocated", 8, true, "  ");
  std::vector<std::string> deallocation = heap_memory_region->generate_deallocation("reallocated", 8, "  ");
  reused_region_canvas_repeat->add_during_lifetime({
    "%c0 = arith.constant 0 : index",
    "%c1 = arith.constant 1 : index",
    "%c_max = arith.constant " + max_reallocated_retries_validation + " : index"
  });
  reused_region_canvas_repeat->add_during_lifetime({//TODO: 这里替换为scf.for的写法
    "%reallocated_for = scf.for %counter = %c0 to %c_max step %c1",
    "    iter_args(%reallocated_iter = %reallocated)",
    "    -> (memref<8xi8>) {",
    "  memref.dealloc %reallocated_iter : memref<8xi8>"
  });

  reused_region_canvas_repeat->add_during_lifetime(allocation);

  std::vector<std::string> repeated_yield = 
    {"  scf.yield %new_reallocated : memref<8xi8>",
     "}"};
  reused_region_canvas_repeat->add_during_lifetime( repeated_yield);
  reused_region_canvas_repeat->add_during_lifetime(access_type_code);
  reused_region_canvas_repeat->add_during_lifetime("func.call @exit(%test_success) : (i32) -> ()");
  reused_region_canvas_repeat->add_variant_description_line("with repeated attempts");
  variants = {reused_region_canvas_simple, reused_region_canvas_repeat};

  return variants;
}

std::vector< std::shared_ptr<RegionCodeCanvas> >UseAfterStar::_generate_reused_mem_stack_validation(
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location) const
{
  std::vector< std::shared_ptr<RegionCodeCanvas> >variants = {};
  CodeCanvas code_simple;
  std::shared_ptr<StackRegion> stack_memory_region = std::dynamic_pointer_cast<StackRegion>(memory_region);
  std::shared_ptr<StackRegion> stack_memory_region_simple = std::make_shared<StackRegion>(*stack_memory_region);
  std::shared_ptr<RegionCodeCanvas> region_canvas = stack_memory_region_simple->generate(std::make_shared<CodeCanvas>(code_simple), "target", 8, true);

  region_canvas->add_to_f_body({
    "%c0_v = arith.constant 0 : index",
    "%c8_v = arith.constant 8 : index",
    "%view_target = memref.view %target[%c0_v][%c8_v] : memref<8xi8> to memref<?xi8>"
  });
  AccessLocation::SplitAccess access_type_code = access_location->generate_split_const_vars(
    access_action, "view_target", 8, 0, "", false, "0", "memref<?xi8>");

  std::shared_ptr<RegionCodeCanvas> reused_region_canvas_simple = stack_memory_region_simple->generate(
    region_canvas, "reallocated", 8, false
  );
  reused_region_canvas_simple->add_to_f_body(access_type_code.to_lines());
  reused_region_canvas_simple->add_to_f_body("func.call @exit(%test_success) : (i32) -> ()");
  variants.push_back( reused_region_canvas_simple );

  std::shared_ptr<RegionCodeCanvas> reused_region_canvas_repeated = stack_memory_region_simple->generate(
    region_canvas, "reallocated", 8, false
    );
  reused_region_canvas_repeated->add_global("memref.global @last_address : memref<8xi8> = dense<0>");
  std::vector<std::string> store_last_address = {"%c0 = arith.constant 0 : index",
    "%last_address = memref.get_global @last_address : memref<8xi8>",
    "%val_realloc = memref.load %reallocated[%c0] : memref<8xi8>",
    "memref.store %val_realloc, %last_address[%c0] : memref<8xi8>"};
  reused_region_canvas_repeated->add_to_f_body(  store_last_address );
  reused_region_canvas_repeated->add_to_f_body(access_type_code.to_lines());
  reused_region_canvas_repeated->add_at(reused_region_canvas_repeated->get_f_call_pos(),
    std::vector<std::string>{//TODO: 这里替换为scf.for的写法
        "%test_success = arith.constant 42 : i32",
      "%c0 = arith.constant 0 : index",
      "%c1 = arith.constant 1 : index",
      "%c_max = arith.constant " + max_reallocated_retries_validation + " : index",
      "%ctrue = arith.constant 1 : i1",
      "%not_matched_for = scf.for %counter = %c0 to %c_max step %c1",
      "    iter_args(%not_matched_iter = %ctrue)",
      "    -> (i1) {",
      "  %next_not_matched = scf.if %not_matched_iter -> (i1) {"
    },
    "  "
  );
  reused_region_canvas_repeated->add_at(reused_region_canvas_repeated->get_f_call_pos() + 1,
    std::vector<std::string>{
      "    %cfalse = arith.constant 0 : i1",
      "    scf.yield %cfalse : i1",
      "  } else {",
      "    scf.yield %not_matched_iter : i1",
      "  }",
      "  scf.yield %next_not_matched : i1",
      "}",
      "func.call @exit(%test_success) : (i32) -> ()"
    },
    "  "
  );
  reused_region_canvas_repeated->add_variant_description_line("with repeated attempts");
  variants.push_back( reused_region_canvas_repeated );

  CodeCanvas code_array;
  std::shared_ptr<StackRegion> stack_memory_region_array = std::make_shared<StackRegion>(*stack_memory_region);
  std::shared_ptr<RegionCodeCanvas> array_region_canvas = stack_memory_region_array->generate_array(std::make_shared<CodeCanvas>(code_array), "target", 8, 16, true);
  array_region_canvas->add_to_f_body({
    "%c0_v = arith.constant 0 : index",
    "%c8_v = arith.constant 8 : index",
    "%view_target = memref.subview %target[0,0][16,8][1,1] : memref<16x8xi8> to memref<16x8xi8>"
  });
  AccessLocation::SplitAccess access_type_code_array = access_location->generate_split_const_vars(
    access_action, "view_target", 8, 16, "c0", false, "0", "memref<16x8xi8>");
  std::shared_ptr<RegionCodeCanvas> reused_region_canvas_simple_array = stack_memory_region_array->generate(
    array_region_canvas, "reallocated", 8, false
  );

  reused_region_canvas_simple_array->add_to_f_body({
    "%c0 = arith.constant 0 : index",
    "%c1 = arith.constant 1 : index",
    "%c8 = arith.constant 8 : index",
    "%c16 = arith.constant 16 : index",
    "%cfalse = arith.constant 0 : i1",
    "%found = scf.for %counter = %c0 to %c16 step %c1 iter_args(%found_iter = %cfalse) -> (i1) {",
    "  %realloc_ptr = memref.extract_aligned_pointer_as_index %reallocated : memref<8xi8> -> index",
    "  %target_base = memref.extract_aligned_pointer_as_index %view_target : memref<16x8xi8> -> index",
    "  %offset = arith.muli %counter, %c8 : index",
    "  %target_ptr = arith.addi %target_base, %offset : index",
    "  %eq = arith.cmpi eq, %realloc_ptr, %target_ptr : index",
    "  %found_next = arith.ori %found_iter, %eq : i1",
    "  scf.yield %found_next : i1",
    "}"
  });
  reused_region_canvas_simple_array->add_to_f_body(access_type_code_array.to_lines());
  reused_region_canvas_simple_array->add_to_f_body("func.call @exit(%test_success) : (i32) -> ()");
  reused_region_canvas_simple_array->add_variant_description_line("using an array of objects");

  variants.push_back( reused_region_canvas_simple_array );
  CodeCanvas code_array_repeated;
  std::shared_ptr<StackRegion> stack_memory_region_array_repeated = std::make_shared<StackRegion>(*stack_memory_region);
  array_region_canvas = stack_memory_region_array_repeated->generate_array(std::make_shared<CodeCanvas>(code_array), "target", 8, 16, true);

  array_region_canvas->add_to_f_body({
    "%c0_v = arith.constant 0 : index",
    "%c8_v = arith.constant 8 : index",
    "%view_target = memref.subview %target[0,0][16,8][1,1] : memref<16x8xi8> to memref<16x8xi8>"
  });  
  AccessLocation::SplitAccess access_type_code_array_repeated = access_location->generate_split_const_vars(
    access_action, "view_target", 8, 16, "c0", false, "0", "memref<16x8xi8>");
  std::shared_ptr<RegionCodeCanvas> reused_region_canvas_array_repeated = stack_memory_region_array->generate(
    array_region_canvas, "reallocated", 8, false
  );
  reused_region_canvas_array_repeated->add_global("memref.global @last_address : memref<8xi8> = dense<0>");

  reused_region_canvas_array_repeated->add_to_f_body({
        "%test_success = arith.constant 42 : i32",
"%c0 = arith.constant 0 : index",
    "%c1 = arith.constant 1 : index",
    "%c8 = arith.constant 8 : index",
    "%c16 = arith.constant 16 : index",
    "%cfalse = arith.constant 0 : i1",
    "%found = scf.for %counter = %c0 to %c16 step %c1 iter_args(%found_iter = %cfalse) -> (i1) {",
    "  %realloc_ptr = memref.extract_aligned_pointer_as_index %reallocated : memref<8xi8> -> index",
    "  %target_base = memref.extract_aligned_pointer_as_index %view_target : memref<16x8xi8> -> index",
    "  %offset = arith.muli %counter, %c8 : index",
    "  %target_ptr = arith.addi %target_base, %offset : index",
    "  %eq = arith.cmpi eq, %realloc_ptr, %target_ptr : index",
    "  %found_next = arith.ori %found_iter, %eq : i1",
    "  scf.yield %found_next : i1",
    "}"
  });
  reused_region_canvas_array_repeated->add_to_f_body(access_type_code_array_repeated.to_lines());
  reused_region_canvas_array_repeated->add_to_f_body("func.call @exit(%test_success) : (i32) -> ()");
  reused_region_canvas_array_repeated->add_at(reused_region_canvas_array_repeated->get_f_call_pos(),
    std::vector<std::string>{//TODO: 这里替换为scf.for的写法
      "%test_success = arith.constant 42 : i32",
      "%c0 = arith.constant 0 : index",
      "%c1 = arith.constant 1 : index",
      "%c_max = arith.constant " + max_reallocated_retries_validation + " : index",
      "%ctrue = arith.constant 1 : i1",
      "%not_matched_for = scf.for %counter = %c0 to %c_max step %c1",
      "    iter_args(%not_matched_iter = %ctrue)",
      "    -> (i1) {",
      "  %next_not_matched = scf.if %not_matched_iter -> (i1) {"
    },
    "  "
  );
  reused_region_canvas_array_repeated->add_at(reused_region_canvas_array_repeated->get_f_call_pos() + 1,
    std::vector<std::string>{
      "    %cfalse = arith.constant 0 : i1",
      "    scf.yield %cfalse : i1",
      "  } else {",
      "    scf.yield %not_matched_iter : i1",
      "  }",
      "  scf.yield %next_not_matched : i1",
      "}",
      "func.call @exit(%test_success) : (i32) -> ()"
    },
    "  "
  );
  reused_region_canvas_array_repeated->add_variant_description_line("repeated attempts");
  reused_region_canvas_array_repeated->add_variant_description_line("using an array of objects");

  variants.push_back( reused_region_canvas_array_repeated );

  return variants;
}
