/*
 * This file is distributed under the Apache License, Version 2.0; refer to
 * LICENSE for details.
 *
 * Initial author: Emanuel Vintila
 */

#include "misuse_of_free.h"

#include "misc.h"
#include "generator/primitives/bug_types/temporal/memory_state/used.h"
#include "generator/primitives/regions/heap_region.h"
#include "generator/primitives/regions/stack_region.h"
#include <string>
#include <vector>

MisuseOfFree::MisuseOfFree():
  TemporalBugType("misuse_of_free")
{
}

bool MisuseOfFree::accepts(std::shared_ptr<MemoryState> memory_state)
{
  return true;
}

bool MisuseOfFree::accepts(std::shared_ptr<Region> region)
{
  return true;
}

bool MisuseOfFree::accepts(std::shared_ptr<AccessLocation> access_location)
{
  return true;
}

std::vector< std::shared_ptr<RegionCodeCanvas> >MisuseOfFree::generate(
  std::shared_ptr<MemoryState> memory_state,
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location
  ) const
{
  std::vector< std::shared_ptr<RegionCodeCanvas> > full_variants;
  /*
  */

  CodeCanvas code;
  code.add_global("func.func @fake_free(%arg0: i8) -> () {func.return}");
  code.add_global("memref.global @heap_obj : memref<8xi8> = uninitialized");
  code.add_test_case_description_line("Memory region: " + memory_region->get_name());
  code.add_test_case_description_line("Bug type: misuse-of-free, " + memory_state->get_printable_name());
  code.add_test_case_description_line("Access type: " + access_location->get_name() + ", " + access_action->get_name());

  std::vector< std::string > magic_values = {"0x20", "0x40", "0x60"};
  std::shared_ptr<RegionCodeCanvas> region_canvas = memory_region->generate(std::make_shared<CodeCanvas>(code), "target", 160, false);

  for ( const std::string &magic_value: magic_values)
  {
    std::shared_ptr<RegionCodeCanvas> region_canvas_with_magic_value = std::make_shared<RegionCodeCanvas>(*region_canvas);

    region_canvas_with_magic_value->add_variant_description_line("magic value " + magic_value);

      region_canvas_with_magic_value->add_during_lifetime({
      "  %c8_mof = arith.constant 8 : index",
      "  %c0 = arith.constant 0 : index",
      "  %c104_mof = arith.constant 104 : index",
      "  %c_magic = arith.constant " + std::to_string(std::stoi(magic_value, nullptr, 16)) + " : i8",
      "  %c0x40 = arith.constant 64 : i8",
      "  memref.store %c_magic, %target[%c8_mof] : memref<160xi8> // magic value",
      "  memref.store %c0x40, %target[%c104_mof] : memref<160xi8>",
      "  %crafted = memref.view %target[%c104_mof][%c8_mof] : memref<160xi8> to memref<?xi8>",
      "  %_ = memref.alloc() : memref<8xi8>",
      "  %craft_val = memref.load %crafted[%c0] : memref<?xi8>",
      "  func.call @fake_free(%craft_val) : (i8) -> ()",
      "",
      "  %heap_obj = memref.get_global @heap_obj : memref<8xi8>"
    });
    std::vector<std::string> access_type_code = access_location->generate(
      access_action, "heap_obj", 8);
    CodeCanvas::code_pos_t index;
    if (is_a<UsedMemory>(memory_state))
    {
      if (std::dynamic_pointer_cast<StackRegion>(memory_region))
      {
        // stack: insert after f() returns in main
        access_type_code.insert(access_type_code.begin(), "%heap_obj = memref.get_global @heap_obj : memref<8xi8>");
        access_type_code.insert(access_type_code.begin(), "%test_success = arith.constant 42 : i32");
        index = region_canvas_with_magic_value->add_at(region_canvas_with_magic_value->get_f_call_pos() + 1, access_type_code, "  ");
      }
      else
      {
        index = region_canvas_with_magic_value->add_at(region_canvas_with_magic_value->get_lifetime_pos(), access_type_code, "  ");
      }
    }
    else
    {
      if ( std::dynamic_pointer_cast<HeapRegion>(memory_region) )
      {
        // unused heap memory
        region_canvas_with_magic_value->add_at(
          region_canvas_with_magic_value->get_deallocation_pos() - 1,
          std::vector<std::string>{
            "%target_ptr = memref.extract_aligned_pointer_as_index %target : memref<160xi8> -> index",
            "%crafted_ptr = memref.extract_aligned_pointer_as_index %crafted : memref<?xi8> -> index",
            "%eq = arith.cmpi eq, %target_ptr, %crafted_ptr : index",
            "%ctrue = arith.constant 1 : i1",
            "%neq = arith.xori %eq, %ctrue : i1",
            "scf.if %neq {"
          },
          "    "
          );
        index = region_canvas_with_magic_value->add_at(
          region_canvas_with_magic_value->get_deallocation_pos(),
          std::vector<std::string>{
            "}"
          },
          "  "
        );
        index = region_canvas_with_magic_value->add_at(index, access_type_code, "    ");
      }
      else
      {
        // stack: insert after f() returns in main
        access_type_code.insert(access_type_code.begin(), "%test_success = arith.constant 42 : i32");
        access_type_code.insert(access_type_code.begin(), "%heap_obj = memref.get_global @heap_obj : memref<8xi8>");
        index = region_canvas_with_magic_value->add_at(region_canvas_with_magic_value->get_f_call_pos() + 1, access_type_code, "  ");
      }
    }
    region_canvas_with_magic_value->add_at(index, "func.call @exit(%test_success) : (i32) -> ()", "  ");
    full_variants.push_back( region_canvas_with_magic_value );
  }

  return full_variants;
}


std::vector< std::shared_ptr<RegionCodeCanvas> >MisuseOfFree::generate_validation(
  std::shared_ptr<MemoryState> memory_state,
  std::shared_ptr<Region> memory_region,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location
  ) const
{
  std::vector< std::shared_ptr<RegionCodeCanvas> > full_variants;
  /*
    
  */

  CodeCanvas code;
  code.add_global("memref.global @heap_obj : memref<8xi8> = uninitialized");
  code.add_test_case_description_line("Memory region: " + memory_region->get_name());
  code.add_test_case_description_line("Bug type: misuse-of-free, " + memory_state->get_printable_name());
  code.add_test_case_description_line("Access type: " + access_location->get_name() + ", " + access_action->get_name());


  std::vector< std::string > magic_values = {"0x20", "0x40", "0x60"};
  std::shared_ptr<RegionCodeCanvas> region_canvas = memory_region->generate(std::make_shared<CodeCanvas>(code), "target", 160, false);

  for ( const std::string &magic_value: magic_values)
  {
    std::shared_ptr<RegionCodeCanvas> region_canvas_with_magic_value = std::make_shared<RegionCodeCanvas>(*region_canvas);

    region_canvas_with_magic_value->add_variant_description_line("magic value " + magic_value);

      region_canvas_with_magic_value->add_during_lifetime({
      "  %c8_mof = arith.constant 8 : index",
      "  %c0 = arith.constant 0 : index",
      "  %c104_mof = arith.constant 104 : index",
      "  %c_magic = arith.constant " + std::to_string(std::stoi(magic_value, nullptr, 16)) + " : i8",
      "  %c0x40 = arith.constant 64 : i8",
      "  memref.store %c_magic, %target[%c8_mof] : memref<160xi8> // magic value",
      "  memref.store %c0x40, %target[%c104_mof] : memref<160xi8>",
      "  %crafted = memref.view %target[%c104_mof][%c8_mof] : memref<160xi8> to memref<?xi8>",
      "  %_ = memref.alloc() : memref<8xi8>",
      "",
      "  %heap_obj = memref.get_global @heap_obj : memref<8xi8>"
    });
    std::vector<std::string> access_type_code = access_location->generate(
      access_action, "heap_obj", 8);
    CodeCanvas::code_pos_t index;
    if (is_a<UsedMemory>(memory_state))
    {
      if (std::dynamic_pointer_cast<StackRegion>(memory_region))
      {
        // stack: insert after f() returns in main
        access_type_code.insert(access_type_code.begin(), "%test_success = arith.constant 42 : i32");
        access_type_code.insert(access_type_code.begin(), "%heap_obj = memref.get_global @heap_obj : memref<8xi8>");
        index = region_canvas_with_magic_value->add_at(region_canvas_with_magic_value->get_f_call_pos() + 1, access_type_code, "  ");
   }
      else
      {
        index = region_canvas_with_magic_value->add_at(region_canvas_with_magic_value->get_lifetime_pos(), access_type_code, "  ");
      }
    }
    else
    {
      if ( std::dynamic_pointer_cast<HeapRegion>(memory_region) )
      {
        index = region_canvas_with_magic_value->add_at(region_canvas_with_magic_value->get_deallocation_pos(), access_type_code, "  ");
      }
      else
      {
       // stack: insert after f() returns in main
        access_type_code.insert(access_type_code.begin(), "%test_success = arith.constant 42 : i32");
        access_type_code.insert(access_type_code.begin(), "%heap_obj = memref.get_global @heap_obj : memref<8xi8>");
        index = region_canvas_with_magic_value->add_at(region_canvas_with_magic_value->get_f_call_pos() + 1, access_type_code, "  ");
      }
    }
    region_canvas_with_magic_value->add_at(index, "func.call @exit(%test_success) : (i32) -> ()", "  ");
    full_variants.push_back( region_canvas_with_magic_value );
  }

  return full_variants;
}
