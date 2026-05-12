/*
 * This file is distributed under the Apache License, Version 2.0; refer to
 * LICENSE for details.
 *
 * Initial author: Emanuel Vintila
 */

#include "type_confusion.h"

#include "misc.h"
#include "generator/primitives/access_types/direct_location.h"
#include "generator/primitives/bug_types/spatial/flow/overflow.h"
#include "generator/primitives/bug_types/spatial/flow/underflow.h"
#include "generator/primitives/bug_types/spatial/origin_target_relation/non_object.h"
#include "generator/primitives/bug_types/spatial/origin_target_relation/intra_object.h"
#include "generator/primitives/access_types/read_action.h"

bool TypeConfusion::accepts(std::shared_ptr<Flow> flow) const
{
  return is_a<Overflow>(flow);
}

bool TypeConfusion::accepts(std::shared_ptr<OriginTargetRelation> origin_target_relation) const
{
  return true;
}

bool TypeConfusion::accepts(std::shared_ptr<AccessLocation> access_location) const
{
  return is_a<DirectLocation>(access_location);
}

std::vector<std::shared_ptr<OriginTargetCodeCanvas>> TypeConfusion::generate(
  std::shared_ptr<Region> origin,
  std::shared_ptr<Region> target,
  std::shared_ptr<OriginTargetRelation> origin_target_relation,
  std::shared_ptr<Flow> flow,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location
) const
{
  std::vector< std::shared_ptr<OriginTargetCodeCanvas> >full_variants;
  /*
    <target, origin, aux_ptr allocations> // aux_ptr points to origin
    <action>(aux_ptr, target) // aux_ptr reaches the target
    <action>(aux_ptr, target_size) // access the target
    return 42;
  */
  CodeCanvas variant;
  variant.add_test_case_description_line("Origin: " + origin->get_name());
  variant.add_test_case_description_line("Target: " + target->get_name());
  variant.add_test_case_description_line("Bug type: " + origin_target_relation->get_printable_name() + ", type confusion OOBA, " + flow->get_name());
  variant.add_test_case_description_line("Access type: " + access_location->get_name() + ", " + access_action->get_name());

  auto generate_preconditions_check_distance = std::bind(&Flow::generate_preconditions_check_distance, flow.get(), std::placeholders::_1);
  auto generate_preconditions_check_in_range = std::bind(&Flow::generate_preconditions_check_in_range, flow.get(), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  std::vector< std::shared_ptr<OriginTargetCodeCanvas> > origin_target_canvases = origin_target_relation->generate(
    variant, origin, 8, target, 8);

  for ( auto &origin_target_canvas : origin_target_canvases )
  {
    if ( origin_target_canvas->get_forces_underflow() ) continue; // skip underflows as they are incompatible with type confusions

    ssize_t static_dist = origin_target_canvas->get_distance_static_value();
    if ( is_a<Overflow>(flow) && static_dist < 0 ) continue;
    if ( is_a<Underflow>(flow) && static_dist > 0 ) continue;

    // manual i32 assembly variant (replaces reinterpret_cast)
    std::shared_ptr<OriginTargetCodeCanvas> variant_manual_i32 = std::make_shared<OriginTargetCodeCanvas>(*origin_target_canvas);
    bool needs_strided = is_a<IntraObject>(origin_target_relation);
    std::string origin_name = variant_manual_i32->get_origin_name();
    std::string dist = variant_manual_i32->get_distance();
    std::string origin_offset = "0";
    if ( is_a<IntraObject>(origin_target_relation) && static_dist < 0 )
      origin_offset = std::to_string(std::abs(static_dist));
    std::string origin_type = needs_strided ? "memref<8xi8, strided<[1], offset: " + origin_offset + ">>" : "memref<8xi8>";

    // big type variant
    std::shared_ptr<OriginTargetCodeCanvas> variant_big_type = std::make_shared<OriginTargetCodeCanvas>(*origin_target_canvas);
    std::vector<std::string> big_type_code = access_location->generate_big_type(
      access_action,
      origin_name,
      origin_type,
      "c0",   // view offset
      "c2",   // view sizes (dynamic dim)
      origin_target_canvas->get_distance()
    );
    variant_big_type->add_during_lifetime(big_type_code);
    variant_big_type->add_during_lifetime("func.call @exit(%test_success) : (i32) -> ()");
    variant_big_type->add_variant_description_line("using memref.view to big type");
    full_variants.push_back(variant_big_type);

    // load widening variant
    std::shared_ptr<OriginTargetCodeCanvas> variant_with_load_widening = std::make_shared<OriginTargetCodeCanvas>(*origin_target_canvas);
    std::vector<std::string> load_widening_code = access_location->generate_load_widening(
      access_action,
      variant_with_load_widening->get_origin_name(),
      origin_type,
      "c4",   // view offset: 4 bytes, causing second i32 to be OOB
      "c1"    // access index: second i32
    );
    variant_with_load_widening->add_during_lifetime(load_widening_code);
    variant_with_load_widening->add_during_lifetime("func.call @exit(%test_success) : (i32) -> ()");
    variant_with_load_widening->add_variant_description_line("using memref.view for load widening");
    full_variants.push_back(variant_with_load_widening);
  }

  return full_variants;
}


std::vector<std::shared_ptr<OriginTargetCodeCanvas>> TypeConfusion::generate_validation(
  std::shared_ptr<Region> origin,
  std::shared_ptr<Region> target,
  std::shared_ptr<OriginTargetRelation> origin_target_relation,
  std::shared_ptr<Flow> flow,
  std::shared_ptr<AccessAction> access_action,
  std::shared_ptr<AccessLocation> access_location
) const
{
  std::vector< std::shared_ptr<OriginTargetCodeCanvas> >full_variants;
  /*
    <target, origin, aux_ptr allocations> // aux_ptr points to origin
    <action>(aux_ptr, target_size) // access the target
    return 42;
  */
  CodeCanvas variant;
  variant.add_test_case_description_line("Origin: " + origin->get_name());
  variant.add_test_case_description_line("Target: " + target->get_name());
  variant.add_test_case_description_line("Bug type: " + origin_target_relation->get_printable_name() + ", type confusion OOBA, " + flow->get_name());
  variant.add_test_case_description_line("Access type: " + access_location->get_name() + ", " + access_action->get_name());

  std::vector< std::shared_ptr<OriginTargetCodeCanvas> > origin_target_canvases = origin_target_relation->generate(
    variant, origin, 8, target, 8);

  for ( auto &origin_target_canvas : origin_target_canvases )
  {
    if ( origin_target_canvas->get_forces_underflow() ) continue; // skip underflows

    ssize_t static_dist = origin_target_canvas->get_distance_static_value();
    // if ( is_a<Overflow>(flow) && static_dist < 0 ) continue;
    // if ( is_a<Underflow>(flow) && static_dist > 0 ) continue;

    bool needs_strided = is_a<IntraObject>(origin_target_relation);

    // big type variant
    std::shared_ptr<OriginTargetCodeCanvas> variant_with_big_type = std::make_shared<OriginTargetCodeCanvas>(*origin_target_canvas);

    std::string var_name_to_access;
    std::string var_offset = "0";
    if ( origin_target_canvas->is_target_allocated() )
    {
      var_name_to_access = variant_with_big_type->get_target_name();
      ssize_t static_dist = origin_target_canvas->get_distance_static_value();
      if ( static_dist > 0 )
        var_offset = std::to_string(static_dist);
    }
    else
    {
      var_name_to_access = variant_with_big_type->get_origin_name();
    }

    std::vector<std::string> access_target_code = access_location->generate_at_index(
      access_action,
      var_name_to_access,
      "c0",
      variant_with_big_type->get_target_size(),
      nullptr,
      needs_strided,
      var_offset
    );
    variant_with_big_type->add_during_lifetime(access_target_code);
    variant_with_big_type->add_during_lifetime("func.call @exit(%test_success) : (i32) -> ()");
    variant_with_big_type->add_variant_description_line("using reinterpret_cast to large memref");
    full_variants.push_back( variant_with_big_type );

    // load widening variant
    std::shared_ptr<OriginTargetCodeCanvas> variant_with_load_widening = std::make_shared<OriginTargetCodeCanvas>(*origin_target_canvas);
    access_target_code = access_location->generate_uint8(
      access_action,
      variant_with_load_widening->get_origin_name(),
      variant_with_load_widening->get_origin_name(),
      variant_with_load_widening->get_distance(),
      8,
      nullptr,
      needs_strided,
      "0"
    );
    variant_with_load_widening->add_during_lifetime(access_target_code);
    variant_with_load_widening->add_during_lifetime("func.call @exit(%test_success) : (i32) -> ()");
    variant_with_load_widening->add_variant_description_line("using load widening");
    full_variants.push_back( variant_with_load_widening );
  }

  return full_variants;
}
