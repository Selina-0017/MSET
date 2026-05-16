/*
 * This file is distributed under the Apache License, Version 2.0; refer to
 * LICENSE for details.
 *
 * Initial author: Emanuel Vintila
 */

#include "inter_object.h"
#include <random>
#include <string>

#include "misc.h"

#ifndef MAX_DISTANCE
#define MAX_DISTANCE 1024
#endif

bool InterObject::accepts(std::shared_ptr<Region> origin, std::shared_ptr<Region> target) const
{
  // accept any combinations
  return true;
}

// static int generate_rand_distance()
// {
//   static std::mt19937 gen(std::random_device{}());
//   std::uniform_int_distribution<> dist(0, MAX_DISTANCE);
//   return dist(gen);
// }

std::vector< std::shared_ptr<OriginTargetCodeCanvas> > InterObject::generate(
  CodeCanvas &canvas,
  std::shared_ptr<Region> origin,
  size_t origin_size,
  std::shared_ptr<Region> target,
  size_t target_size
) const
{
  std::vector< std::shared_ptr<OriginTargetCodeCanvas> > variants;
  std::shared_ptr<CodeCanvas> canvas_ptr = std::make_shared<CodeCanvas>(canvas);
  std::shared_ptr<RegionCodeCanvas> origin_canvas;
  std::shared_ptr<RegionCodeCanvas> target_canvas;
  // int rand_distance = generate_rand_distance();

  // variant 1: origin first, target second (distance positive)
  origin_canvas = origin->generate( canvas_ptr, "origin", origin_size, true  );
  target_canvas = target->generate( origin_canvas->get_lifetime_pos(), origin_canvas, "target", target_size, true  );
  std::string distance = "distance";
  std::string distance_negated = "distance_negated";

  target_canvas->add_locals(
    {
    "%ptr_origin = memref.extract_aligned_pointer_as_index %origin : memref<8xi8> -> index",
    "%ptr_target = memref.extract_aligned_pointer_as_index %target : memref<8xi8> -> index",
    "%c0 = arith.constant 0 : index",
  });
    // distance = ptr_target - ptr_origin
  // target_canvas->add_locals({
  //   "%diff = arith.subi %ptr_target, %ptr_origin : index",
  //   "%minus_diff = arith.subi %c0, %diff : index",
  //   "%is_pos = arith.cmpi sgt, %diff, %c0 : index",
  // });
  
  target_canvas->add_locals(
    {
    // "%distance = arith.select %is_pos, %diff, %minus_diff : index",
    "%distance = arith.subi %ptr_target, %ptr_origin : index",
    "%temp = arith.subi %ptr_origin, %ptr_target : index",
    "%distance_negated = arith.subi %c0, %temp : index"
    }
  );

  std::shared_ptr<OriginTargetCodeCanvas> variant = std::make_shared<OriginTargetCodeCanvas>( target_canvas, target_size, origin_size, "target", "origin", distance, distance_negated );
  variant->set_lifetime_pos( target_canvas->get_lifetime_pos() );
  variant->add_variant_description_line("target declared after origin");
  variants.push_back(variant);

  // variant 2: target first, origin second (distance negative)
  target_canvas = target->generate(canvas_ptr, "target", target_size, true);
  origin_canvas = origin->generate(target_canvas->get_lifetime_pos(), target_canvas, "origin", origin_size, true);
  origin_canvas->add_locals(
{
    "%ptr_target = memref.extract_aligned_pointer_as_index %target : memref<8xi8> -> index",
    "%ptr_origin = memref.extract_aligned_pointer_as_index %origin : memref<8xi8> -> index"
      });
    // distance = ptr_ - ptr_target
  // origin_canvas->add_locals({
  //   "%diff = arith.subi %ptr_, %ptr_target : index",
  //   "%minus_diff = arith.subi %c0, %diff : index",
  //   "%is_pos = arith.cmpi sgt, %diff, %c0 : index",
  // });
  
  origin_canvas->add_locals({
    // "%distance = arith.select %is_pos, %diff, %minus_diff : index",
    "%distance = arith.subi %ptr_target, %ptr_origin : index",
    "%temp = arith.subi %ptr_origin, %ptr_target : index",
    "%distance_negated = arith.subi %c0, %temp : index"
  });

  variant = std::make_shared<OriginTargetCodeCanvas>( origin_canvas, target_size, origin_size, "target", "origin", distance, distance_negated );
  variant->set_lifetime_pos( origin_canvas->get_lifetime_pos());
  variant->add_variant_description_line("target declared before origin");
  variants.push_back(variant);

  return variants;
}
