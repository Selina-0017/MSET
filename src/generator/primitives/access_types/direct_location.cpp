/*
 * This file is distributed under the Apache License, Version 2.0; refer to
 * LICENSE for details.
 *
 * Initial author: Emanuel Vintila
 */

#include "direct_location.h"

#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "misc.h"
#include "generator/primitives/access_types/read_action.h"
#include "generator/primitives/bug_types/spatial/flow/flow.h"

// simple generate
std::vector<std::string> DirectLocation::generate(std::shared_ptr<AccessAction> action, const std::string &access_var_name, size_t size, size_t array_size, const std::string &index_var, bool needs_strided, const std::string &offset, const std::string &custom_type) const
{
  return generate_split_const_vars(action, access_var_name, size, array_size, index_var, needs_strided, offset, custom_type).to_lines();
}

// simple split, using auxiliary size and content variables
AccessLocation::SplitAccess DirectLocation::generate_split_aux_vars(
  std::shared_ptr<AccessAction> action,
  const std::string &access_var_name,
  size_t size,
  size_t array_size,
  const std::string &index_var,
  std::function<std::vector<std::string>(const std::string&)> generate_counter_update,
  const std::string &distance,
  bool needs_strided,
  const std::string &offset
) const
{
  SplitAccess split_access;
  std::string size_str = std::to_string(size);
  std::string stride_suffix = needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">";

  std::string distance_expr = distance;
  std::string line = "";
  // if(needs_strided && distance.find("negated")!= std::string::npos){
  //   distance_expr = "negadist_variant2";
  //   line = "%" + distance_expr + " = arith.subi %c0, %" + distance + " : index";
  // }

  if (is_a<ReadAction>(action))
  {
    // READ
    split_access.aux_variables = {
      {"c0", "index", "", "0"},
      {"c1", "index", "", "1"},
      {"c" + size_str, "index", "", size_str},
    };

    if (array_size > 0) {
      std::string array_size_str = std::to_string(array_size);
      split_access.access_lines.emplace_back("scf.for %i = %c0 to %c" + size_str + " step %c1 {");
      {
        std::string idx_expr = "%" + index_var + ", %i";
        if (!offset.empty() && offset != "0" && !needs_strided) {
          split_access.access_lines.emplace_back("  %__base = arith.constant " + offset + " : index");
          split_access.access_lines.emplace_back("  %__idx = arith.addi %" + index_var + ", %__base : index");
          idx_expr = "%__idx, %i";
        }
        split_access.access_lines.emplace_back("  %val = memref.load %" + access_var_name + "[" + idx_expr + "] : memref<" + array_size_str + "x" + size_str + "xi8" + stride_suffix);
        split_access.access_lines.emplace_back("  func.call @use(%val) : (i8) -> ()");
      }
      split_access.access_lines.emplace_back("}");
    } else {
      split_access.access_lines.emplace_back(line);
      split_access.access_lines.emplace_back("scf.for %i = %c0 to %c" + size_str + " step %c1 {");
      
        // if (!offset.empty() && offset != "0" && !needs_strided) {
        //   split_access.access_lines.emplace_back("  %__base = arith.constant " + offset + " : index");
        //   split_access.access_lines.emplace_back("  %__idx = arith.addi %i, %__base : index");
        //   idx_expr = "%__idx";
        // }

      split_access.access_lines.emplace_back("  %idx = arith.addi %i, %" + distance_expr + " : index");
      split_access.access_lines.emplace_back("  %val = memref.load %" + access_var_name + "[%idx] : memref<" + size_str + "xi8" + stride_suffix);
      split_access.access_lines.emplace_back("  func.call @use(%val) : (i8) -> ()");
      split_access.access_lines.emplace_back("}");
    }
  }
  else
  {
    // WRITE
    split_access.aux_variables = {
      {"c0", "index", "", "0"},
      {"c1", "index", "", "1"},
      {"c" + size_str, "index", "", size_str},
      {"c0xFF", "i8", "", "255"},
    };
    if (array_size > 0) {
      std::string array_size_str = std::to_string(array_size);
      split_access.access_lines.emplace_back("scf.for %i = %c0 to %c" + size_str + " step %c1 {");
      {
        std::string idx_expr = "%" + index_var + ", %i";
        if (!offset.empty() && offset != "0" && !needs_strided) {
          split_access.access_lines.emplace_back("  %__base = arith.constant " + offset + " : index");
          split_access.access_lines.emplace_back("  %__idx = arith.addi %" + index_var + ", %__base : index");
          idx_expr = "%__idx, %i";
        }
        split_access.access_lines.emplace_back("  memref.store %c0xFF, %" + access_var_name + "[" + idx_expr + "] : memref<" + array_size_str + "x" + size_str + "xi8" + stride_suffix);
      }
      split_access.access_lines.emplace_back("}");
    } else {
      split_access.access_lines.emplace_back(line);
      split_access.access_lines.emplace_back("scf.for %i = %c0 to %c" + size_str + " step %c1 {");
      
        // if (!offset.empty() && offset != "0" && !needs_strided) {
        //   split_access.access_lines.emplace_back("  %__base = arith.constant " + offset + " : index");
        //   split_access.access_lines.emplace_back("  %__idx = arith.addi %i, %__base : index");
        //   idx_expr = "%__idx";
        // }
      split_access.access_lines.emplace_back("  %idx = arith.addi %i, %" + distance_expr + " : index");
      split_access.access_lines.emplace_back("  memref.store %c0xFF, %" + access_var_name + "[%idx] : memref<" + size_str + "xi8" + stride_suffix);
      split_access.access_lines.emplace_back("}");
    }
  }
  split_access.description = "auxiliary variables";
  return split_access;
}

// simple split, using const size and content variables
AccessLocation::SplitAccess DirectLocation::generate_split_const_vars(
  std::shared_ptr<AccessAction> action,
  const std::string &access_var_name,
  size_t size,
  size_t array_size,
  const std::string &index_var,
  bool needs_strided,
  const std::string &offset,
  const std::string &custom_type
) const
{
  SplitAccess split_access;
  std::string size_str = std::to_string(size);
  std::string stride_suffix = needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">";
  std::string base_type;
  if (!custom_type.empty()) {
    base_type = custom_type;
  } else {
    base_type = "memref<" + size_str + (array_size > 0 ? "x" + std::to_string(array_size) : "") + "xi8" + stride_suffix;
  }
  if (is_a<ReadAction>(action))
  {
    // READ
    split_access.aux_variables = {
      {"c0", "index", "", "0"},
      {"c1", "index", "", "1"},
      {"c" + size_str, "index", "", size_str},
    };

    if (array_size > 0) {
      std::string array_size_str = std::to_string(array_size);
      split_access.access_lines.emplace_back("scf.for %i = %c0 to %c" + size_str + " step %c1 {");
      {
        std::string idx_expr = "%" + index_var + ", %i";
        if (!offset.empty() && offset != "0" && !needs_strided) {
          split_access.access_lines.emplace_back("  %__base = arith.constant " + offset + " : index");
          split_access.access_lines.emplace_back("  %__idx = arith.addi %" + index_var + ", %__base : index");
          idx_expr = "%__idx, %i";
        }
        split_access.access_lines.emplace_back("  %val = memref.load %" + access_var_name + "[" + idx_expr + "] : " + base_type);
        split_access.access_lines.emplace_back("  func.call @use(%val) : (i8) -> ()");
      }
      split_access.access_lines.emplace_back("}");
    } else {
      split_access.access_lines.emplace_back("scf.for %i = %c0 to %c" + size_str + " step %c1 {");
      {
        std::string idx_expr = "%i";
        if (!offset.empty() && offset != "0" && !needs_strided) {
          split_access.access_lines.emplace_back("  %__base = arith.constant " + offset + " : index");
          split_access.access_lines.emplace_back("  %__idx = arith.addi %i, %__base : index");
          idx_expr = "%__idx";
        }
        split_access.access_lines.emplace_back("  %val = memref.load %" + access_var_name + "[" + idx_expr + "] : " + base_type);
        split_access.access_lines.emplace_back("  func.call @use(%val) : (i8) -> ()");
      }
      split_access.access_lines.emplace_back("}");
    }
  }
  else
  {
    // WRITE
    split_access.aux_variables = {
      {"c0", "index", "", "0"},
      {"c1", "index", "", "1"},
      {"c" + size_str, "index", "", size_str},
      {"c0xFF", "i8", "", "255"},
    };
    if (array_size > 0) {
      std::string array_size_str = std::to_string(array_size);
      split_access.access_lines.emplace_back("scf.for %i = %c0 to %c" + size_str + " step %c1 {");
      {
        std::string idx_expr = "%" + index_var + ", %i";
        if (!offset.empty() && offset != "0" && !needs_strided) {
          split_access.access_lines.emplace_back("  %__base = arith.constant " + offset + " : index");
          split_access.access_lines.emplace_back("  %__idx = arith.addi %" + index_var + ", %__base : index");
          idx_expr = "%__idx, %i";
        }
        split_access.access_lines.emplace_back("  memref.store %c0xFF, %" + access_var_name + "[" + idx_expr + "] : " + base_type);
      }
      split_access.access_lines.emplace_back("}");
    } else {
      split_access.access_lines.emplace_back("scf.for %i = %c0 to %c" + size_str + " step %c1 {");
      {
        std::string idx_expr = "%i";
        if (!offset.empty() && offset != "0" && !needs_strided) {
          split_access.access_lines.emplace_back("  %__base = arith.constant " + offset + " : index");
          split_access.access_lines.emplace_back("  %__idx = arith.addi %i, %__base : index");
          idx_expr = "%__idx";
        }
        split_access.access_lines.emplace_back("  memref.store %c0xFF, %" + access_var_name + "[" + idx_expr + "] : " + base_type);
      }
      split_access.access_lines.emplace_back("}");
    }
  }
  split_access.description = "constants";
  return split_access;
}

// generate from the given index to index + size
std::vector<std::string> DirectLocation::generate_at_index(
  std::shared_ptr<AccessAction> action,
  const std::string &access_var_name,
  std::string index,
  size_t size,
  std::function<std::vector<std::string>(const std::string&)> generate_preconditions_check_distance,
  bool needs_strided,
  const std::string &offset
) const
{
  std::vector<std::string> lines;
  std::string size_str = std::to_string(size);
  std::string stride_suffix = needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">";
  if (is_a<ReadAction>(action))
  {
    // READ
    lines = {
      "%c0 = arith.constant 0 : index",
      "%c1 = arith.constant 1 : index",
      "%c" + size_str + " = arith.constant " + size_str + " : index",
      "scf.for %j = %c0 to %c" + size_str + " step %c1 {",
      "  %idx = arith.addi %j, %" + index + " : index",
      "  %val = memref.load %" + access_var_name + "[%idx] : memref<" + size_str + "xi8" + stride_suffix,
      "  func.call @use(%val) : (i8) -> ()",
      "}",
    };
    if (generate_preconditions_check_distance)
    {
      auto preconds = generate_preconditions_check_distance(index);
      lines.insert(lines.begin(), preconds.begin(), preconds.end());
    }
  }
  else
  {
    // WRITE
    lines = {
      "%c0 = arith.constant 0 : index",
      "%c1 = arith.constant 1 : index",
      "%c" + size_str + " = arith.constant " + size_str + " : index",
      "%c0xFF = arith.constant 255 : i8",
      "scf.for %j = %c0 to %c" + size_str + " step %c1 {",
      "  %idx = arith.addi %j, %" + index + " : index",
      "  memref.store %c0xFF, %" + access_var_name + "[%idx] : memref<" + size_str + "xi8" + stride_suffix,
      "}",
    };
    if (generate_preconditions_check_distance)
    {
      auto preconds = generate_preconditions_check_distance(index);
      lines.insert(lines.begin(), preconds.begin(), preconds.end());
    }
  }
  return lines;
}

// generate using the given index up to index + distance
std::vector<std::string> DirectLocation::generate_using_runtime_index(
  std::shared_ptr<AccessAction> action,
  const std::string &access_var_name,
  std::string index,
  std::string distance,
  std::function<std::vector<std::string>(const std::string&)> generate_preconditions_check_distance,
  std::function<std::vector<std::string>(const std::string&, const std::string&, const std::string&)> generate_preconditions_check_in_range
) const
{
  std::vector<std::string> lines;
  if (is_a<ReadAction>(action))
  {
    // READ
    lines = {
      "%c0 = arith.constant 0 : index",
      "%c1 = arith.constant 1 : index",
      "scf.for %" + index + " = %c0 to %" + distance + " step %c1 {",
      "  %val = memref.load %" + access_var_name + "[%" + index + "] : memref<8xi8>",
      "  func.call @use(%val) : (i8) -> ()",
      "}",
    };
    //TODO
    if (!distance.empty() && generate_preconditions_check_distance) 
    {
      auto preconds = generate_preconditions_check_distance(distance);
      lines.insert(lines.begin(), preconds.begin(), preconds.end());
    }
  }
  else
  {
    // WRITE
    lines = {
      "%c0 = arith.constant 0 : index",
      "%c1 = arith.constant 1 : index",
      "%c0xFF = arith.constant 255 : i8",
      "scf.for %" + index + " = %c0 to %" + distance + " step %c1 {",
      "  memref.store %c0xFF, %" + access_var_name + "[%" + index + "] : memref<8xi8>",
      "}",
    };

    if (!distance.empty())
    {
      // if (generate_preconditions_check_in_range) lines.insert(lines.begin(), "if ( " + generate_preconditions_check_in_range(index, "&" + access_var_name + "[0]", "&" + access_var_name + "[" + distance + "]") + " ) _exit(PRECONDITIONS_FAILED_VALUE);");
      if (generate_preconditions_check_distance)
      {
        auto preconds = generate_preconditions_check_distance(distance);
        lines.insert(lines.begin(), preconds.begin(), preconds.end());
      }
    }//TODO
  }
  return lines;
}

// generate in bulks using an auxiliary pointer
AccessLocation::SplitAccess DirectLocation::generate_bulk_split_using_index(
  std::shared_ptr<AccessAction> action,
  std::string from,
  std::string to,
  std::string distance,
  std::function<std::vector<std::string>(const std::string&)>  generate_preconditions_check_distance,
  std::function<std::vector<std::string>(const std::string&, const std::string&, const std::string&)> generate_preconditions_check_in_range,
  std::function<std::vector<std::string>(const std::string&)> generate_counter_update,
  bool needs_strided,
  const std::string &offset
) const
{
  bool is_underflow = false;
  SplitAccess split_access;
  std::vector<std::string> counter_update = generate_counter_update("reach_index");
  
  for (const auto& line : counter_update)
  {
    if (line.find("subi") != std::string::npos)
    {
      is_underflow = true;
      break;
    }
  }
  std::string index_var = is_underflow ? "index" : "reach_index";
  std::string appendlines = is_underflow ? "  %index = arith.subi %c0, %reach_index : index" : "";
  std::string dist = (is_number(distance) && std::stoll(distance) == 0) ? "c" + distance : distance;
  std::string stride_suffix = needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">";
  std::string negadist_val;
  if(is_underflow) {
    negadist_val = "%negadist_variant = arith.subi %c0, %" + dist + " : index";
    dist = "negadist_variant";
  }
  if (is_a<ReadAction>(action))
  {
    // READ
    split_access.aux_variables = {
      {"c0", "index", "", "0"},
      {"c1", "index", "", "1"},
    };

    split_access.result = from;
    split_access.access_lines.emplace_back(negadist_val);
    split_access.access_lines.emplace_back("scf.for %reach_index = %c0 to %" + dist + " step %c1 {");
    if (!appendlines.empty()) split_access.access_lines.emplace_back(appendlines);
    
    std::string idx_expr = "%" + index_var;
    split_access.access_lines.emplace_back("  %val = memref.load %" + from + "[" + idx_expr + "] : memref<8xi8" + stride_suffix);
    split_access.access_lines.emplace_back("  func.call @use(%val) : (i8) -> ()");
    
    split_access.access_lines.emplace_back("}");

    if (!distance.empty() && generate_preconditions_check_distance)
    {
      auto preconds = generate_preconditions_check_distance(distance);
      split_access.access_lines.insert(split_access.access_lines.begin(), preconds.begin(), preconds.end());
    } //TODO
  }
  else
  {
    // WRITE
    split_access.aux_variables = {
      {"c0", "index", "", "0"},
      {"c1", "index", "", "1"},
      {"c0xFF", "i8", "", "255"},
    };
    split_access.result = from;
    split_access.access_lines.emplace_back(negadist_val);
    split_access.access_lines.emplace_back("scf.for %reach_index = %c0 to %" + dist + " step %c1 {");
    if (!appendlines.empty()) split_access.access_lines.emplace_back(appendlines);
    {
      std::string idx_expr = "%" + index_var;
      split_access.access_lines.emplace_back("  memref.store %c0xFF, %" + from + "[" + idx_expr + "] : memref<8xi8" + stride_suffix);
    }
    split_access.access_lines.emplace_back("}");
    if (!distance.empty() && generate_preconditions_check_distance)
    {
      auto preconds = generate_preconditions_check_distance(distance);
      split_access.access_lines.insert(split_access.access_lines.begin(), preconds.begin(), preconds.end());
    } //TODO
  }
  split_access.description = "index";
  return split_access;
}

// generate in bulks
AccessLocation::SplitAccess DirectLocation::generate_bulk_split_using_aux_ptr(
  std::shared_ptr<AccessAction> action,
  std::string from,
  std::string to,
  std::string distance,
  std::function<std::vector<std::string>(const std::string&)>  generate_preconditions_check_distance,
  std::function<std::vector<std::string>(const std::string&, const std::string&, const std::string&)>  generate_preconditions_check_in_range,
  std::function<std::vector<std::string>(const std::string&)>  generate_counter_update,
  bool needs_strided,
  const std::string &offset
) const
{
  bool is_underflow = false;
  SplitAccess split_access;
  std::vector<std::string> counter_update = generate_counter_update("reach_index");
  for (const auto& line : counter_update)
  {
    if (line.find("subi") != std::string::npos)
    {
      is_underflow = true;
      break;
    }
  }
  std::string index_var = is_underflow ? "index" : "reach_index";
  std::string appendlines = is_underflow ? "  %index = arith.subi %c0, %reach_index : index" : "";
  std::string dist = (is_number(distance) && std::stoll(distance) == 0) ? "c" + distance : distance;
  std::string stride_suffix = needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">";
  if (is_a<ReadAction>(action))
  {
    // READ
    split_access.aux_variables = {
      {"c0", "index", "", "0"},
      {"c1", "index", "", "1"},
    };

    split_access.result = from;
    split_access.access_lines.emplace_back("scf.for %reach_index = %c0 to %" + dist + " step %c1 {");
    if (!appendlines.empty()) split_access.access_lines.emplace_back(appendlines);
    split_access.access_lines.emplace_back("  %val = memref.load %" + from + "[%" + index_var + "] : memref<8xi8" + stride_suffix);
    split_access.access_lines.emplace_back("  func.call @use(%val) : (i8) -> ()");
    split_access.access_lines.emplace_back("}");
    if (!distance.empty())
    {
      // if (generate_preconditions_check_in_range) split_access.access_lines.insert(split_access.access_lines.begin(), "if ( " + generate_preconditions_check_in_range("aux_ptr", from, to) + " ) _exit(PRECONDITIONS_FAILED_VALUE);");
      if (generate_preconditions_check_distance)
      {
        auto preconds = generate_preconditions_check_distance(distance);
        split_access.access_lines.insert(split_access.access_lines.begin(), preconds.begin(), preconds.end());
      }
    }//TODO
  }
  else
  {
    // WRITE
    split_access.aux_variables = {
      {"c0", "index", "", "0"},
      {"c1", "index", "", "1"},
      {"c0xFF", "i8", "", "255"},
    };
    split_access.result = from;
    split_access.access_lines.emplace_back("scf.for %reach_index = %c0 to %" + dist + " step %c1 {");
    if (!appendlines.empty()) split_access.access_lines.emplace_back(appendlines);
    split_access.access_lines.emplace_back("  memref.store %c0xFF, %" + from + "[%" + index_var + "] : memref<8xi8" + stride_suffix);
    split_access.access_lines.emplace_back("}");
    if (!distance.empty())
    {
      // if (generate_preconditions_check_in_range) split_access.access_lines.insert(split_access.access_lines.begin(), "if ( " + generate_preconditions_check_in_range("aux_ptr", from, to) + " ) _exit(PRECONDITIONS_FAILED_VALUE);");
      if (generate_preconditions_check_distance)
      {
        auto preconds = generate_preconditions_check_distance(distance);
        split_access.access_lines.insert(split_access.access_lines.begin(), preconds.begin(), preconds.end());
      }
    }//TODO
  }
  split_access.description = "auxiliary pointer";
  return split_access;
}

// generate using a load widening to uint32
std::vector<std::string> DirectLocation::generate_uint32(
  std::shared_ptr<AccessAction> action,
  std::string from,
  std::string to,
  std::string distance,
  size_t size,
  std::function<std::vector<std::string>(const std::string&)>  generate_preconditions_check_distance,
  bool needs_strided,
  const std::string &offset
) const
{
  std::vector<std::string> lines;
  std::string size_str = std::to_string(size);
  std::string byte_offset = std::to_string(size - 3);
  std::string stride_suffix = needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">";
  if (is_a<ReadAction>(action))
  {
    // READ: manually assemble 4 i8 bytes into i32
    lines = {
      "%c0 = arith.constant 0 : index",
      "%c1 = arith.constant 1 : index",
      "%c2 = arith.constant 2 : index",
      "%c3 = arith.constant 3 : index",
      "%c8_i32  = arith.constant 8 : i32",
      "%c16_i32 = arith.constant 16 : i32",
      "%c24_i32 = arith.constant 24 : i32",
      "%base = arith.constant " + byte_offset + " : index",
      "%b0 = memref.load %" + from + "[%base] : memref<" + size_str + "xi8" + stride_suffix,
      "func.call @use(%b0) : (i8) -> ()",
      "%idx1 = arith.addi %base, %c1 : index",
      "%b1 = memref.load %" + from + "[%idx1] : memref<" + size_str + "xi8" + stride_suffix,
      "func.call @use(%b1) : (i8) -> ()",
      "%idx2 = arith.addi %base, %c2 : index",
      "%b2 = memref.load %" + from + "[%idx2] : memref<" + size_str + "xi8" + stride_suffix,
      "func.call @use(%b2) : (i8) -> ()",
      "%idx3 = arith.addi %base, %c3 : index",
      "%b3 = memref.load %" + from + "[%idx3] : memref<" + size_str + "xi8" + stride_suffix,
      "func.call @use(%b3) : (i8) -> ()",
      "%b0_i32 = arith.extui %b0 : i8 to i32",
      "%b1_i32 = arith.extui %b1 : i8 to i32",
      "%b2_i32 = arith.extui %b2 : i8 to i32",
      "%b3_i32 = arith.extui %b3 : i8 to i32",
      "%b1_sh = arith.shli %b1_i32, %c8_i32  : i32",
      "%b2_sh = arith.shli %b2_i32, %c16_i32 : i32",
      "%b3_sh = arith.shli %b3_i32, %c24_i32 : i32",
      "%acc0 = arith.ori %b0_i32, %b1_sh : i32",
      "%acc1 = arith.ori %b2_sh, %b3_sh : i32",
      "%val = arith.ori %acc0, %acc1 : i32",
    };
  }
  else
  {
    // WRITE: manually decompose i32 into 4 i8 bytes
    lines = {
      "%c0 = arith.constant 0 : index",
      "%c1 = arith.constant 1 : index",
      "%c2 = arith.constant 2 : index",
      "%c3 = arith.constant 3 : index",
      "%c8_i32  = arith.constant 8 : i32",
      "%c16_i32 = arith.constant 16 : i32",
      "%c24_i32 = arith.constant 24 : i32",
      "%c0xFFFFFFFF = arith.constant 4294967295 : i32",
      "%base = arith.constant " + std::to_string(size - 1) + " : index",
      "%b0 = arith.trunci %c0xFFFFFFFF : i32 to i8",
      "%w1_tmp = arith.shrsi %c0xFFFFFFFF, %c8_i32  : i32",
      "%w2_tmp = arith.shrsi %c0xFFFFFFFF, %c16_i32 : i32",
      "%w3_tmp = arith.shrsi %c0xFFFFFFFF, %c24_i32 : i32",
      "%b1 = arith.trunci %w1_tmp : i32 to i8",
      "%b2 = arith.trunci %w2_tmp : i32 to i8",
      "%b3 = arith.trunci %w3_tmp : i32 to i8",
      "memref.store %b0, %" + from + "[%base] : memref<" + size_str + "xi8" + stride_suffix,
      "%idx1 = arith.addi %base, %c1 : index",
      "memref.store %b1, %" + from + "[%idx1] : memref<" + size_str + "xi8" + stride_suffix,
      "%idx2 = arith.addi %base, %c2 : index",
      "memref.store %b2, %" + from + "[%idx2] : memref<" + size_str + "xi8" + stride_suffix,
      "%idx3 = arith.addi %base, %c3 : index",
      "memref.store %b3, %" + from + "[%idx3] : memref<" + size_str + "xi8" + stride_suffix,
    };
  }
  return lines;
}

// generate after casting to uint8
std::vector<std::string> DirectLocation::generate_uint8(
  std::shared_ptr<AccessAction> action,
  std::string from,
  std::string to,
  std::string distance,
  size_t size,
  std::function<std::vector<std::string>(const std::string&)>  generate_preconditions_check_distance,
  bool needs_strided,
  const std::string &offset
) const
{
  std::vector<std::string> lines;
  std::string size_str = std::to_string(size);
  std::string stride_suffix = needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">";
  if (is_a<ReadAction>(action))
  {
    // READ
    lines = {
      "%idx = arith.constant " + std::to_string(size - 1) + " : index",
      "%val = memref.load %" + from + "[%idx] : memref<" + size_str + "xi8" + stride_suffix,
      "func.call @use(%val) : (i8) -> ()",
    };
  }
  else
  {
    // WRITE
    lines = {
      "%idx = arith.constant " + std::to_string(size - 1) + " : index",
      "%c0xFF = arith.constant 255 : i8",
      "memref.store %c0xFF, %" + from + "[%idx] : memref<" + size_str + "xi8" + stride_suffix,
    };
  }
  return lines;
}

std::vector<std::string> DirectLocation::generate_big_type(
std::shared_ptr<AccessAction> action,
    const std::string &orig_var_name,
    const std::string &orig_type,
    const std::string &view_sizes,
    const std::string &distance,
    std::function<std::vector<std::string>(const std::string&)> generate_preconditions_check_distance,
    const std::string &orig_offset
) const
{
  std::vector<std::string> lines;
  std::string dst_type = "memref<?xi8>";
  if(orig_offset != "0") {
    dst_type = "memref<?xi8, strided<[1], offset: " + orig_offset + ">>";  
  } 
  lines.emplace_back("%viewed = memref.subview %" + orig_var_name + "[0][%" + view_sizes + "][1] : " + orig_type + " to " + dst_type);
  lines.emplace_back("%c0 = arith.constant 0 : index");
  lines.emplace_back("%c1 = arith.constant 1 : index");
  lines.emplace_back("%c8 = arith.constant 8 : index");
  if (is_a<ReadAction>(action))
  {
    lines.emplace_back("scf.for %i = %c0 to %distance step %c1 {");
    lines.emplace_back("  %val = memref.load %viewed[%i] : " + dst_type);
    lines.emplace_back("  func.call @use(%val) : (i8) -> ()");
    lines.emplace_back("}");
    lines.emplace_back("scf.for %j = %c0 to %c8 step %c1 {");
    lines.emplace_back("  %idx = arith.addi %j, %distance : index");
    lines.emplace_back("  %val2 = memref.load %viewed[%idx] : " + dst_type);
    lines.emplace_back("  func.call @use(%val2) : (i8) -> ()");
    lines.emplace_back("}");
  }
  else
  {
    lines.emplace_back("%c0xFF = arith.constant 255 : i8");
    lines.emplace_back("scf.for %i = %c0 to %distance step %c1 {");
    lines.emplace_back("  memref.store %c0xFF, %viewed[%i] : " + dst_type);
    lines.emplace_back("}");
    lines.emplace_back("scf.for %j = %c0 to %c8 step %c1 {");
    lines.emplace_back("  %idx = arith.addi %j, %distance : index");
    lines.emplace_back("  memref.store %c0xFF, %viewed[%idx] : " + dst_type);
    lines.emplace_back("}");
  }
  //插入地址比较
  if (!distance.empty() && generate_preconditions_check_distance)
  {
    std::vector<std::string> precond1 = {
      "%cond11 = arith.cmpi sgt, %" + distance + ", %c0 : index",
      "%cond12 = arith.cmpi sgt, %" + distance + ", %" + view_sizes + " : index",
      "%minus_big = arith.subi %c0, %" + view_sizes + " : index",
      "%cond21 = arith.cmpi slt, %" + distance + ", %c0 : index",
      "%cond22 = arith.cmpi slt, %" + distance + ", %minus_big : index",
      "%cond1 = arith.andi %cond11, %cond12 : i1",
      "%cond2 = arith.andi %cond21, %cond22 : i1",
      "%big_not_enough = arith.ori %cond1, %cond2 : i1",
      "scf.if %big_not_enough {",
      "  func.call @exit(%precond_fail) : (i32) -> ()",
     "  scf.yield",
     "}"
    };
    lines.insert(lines.begin(), precond1.begin(), precond1.end());
    auto precond2 = generate_preconditions_check_distance(distance);
    lines.insert(lines.begin(), precond2.begin(), precond2.end());
  }//TODO
  return lines;
}

std::vector<std::string> DirectLocation::generate_load_widening(
  std::shared_ptr<AccessAction> action,
  const std::string &orig_var_name,
  const std::string &orig_type,
  const std::string &view_offset,
  const std::string &access_index,
  std::function<std::vector<std::string>(const std::string&)> generate_preconditions_check_distance,
  const std::string &distance,
  const std::string &orig_offset
) const
{
  std::vector<std::string> lines;
  lines.emplace_back("%c0 = arith.constant 0 : index");
  lines.emplace_back("%c1 = arith.constant 1 : index");
  lines.emplace_back("%c2 = arith.constant 2 : index");
  lines.emplace_back("%c4 = arith.constant 4 : index");
  // 对 alloc而来 /subview里offset 为0 的origin做view
  std::string dst_type = "memref<2xi32>";
  std::vector<std::string>view_lines ={ "%viewed = memref.view %" + orig_var_name + "[%" + view_offset + "][] : " + orig_type + " to " + dst_type };
  //处理offset != 0 && 由subview 而来的origin。先对parent 做view，再取小的subview。
  if(orig_offset != "0"){
    std::string new_orig_offset = std::to_string((std::stoll(orig_offset)/4));
    view_lines = {
      "%view_s = memref.view %s[ %" + view_offset + "][] : memref<16xi8> to memref<4xi32>",
      "%viewed = memref.subview %view_s[" + new_orig_offset + "][2][1] : memref<4xi32> to memref<2xi32, strided<[1], offset: " + new_orig_offset + ">>"
    };
    dst_type = "memref<2xi32, strided<[1], offset: " + new_orig_offset + ">>";
  }
  lines.insert(lines.end(), view_lines.begin(), view_lines.end());
  if (is_a<ReadAction>(action))
  {
    lines.emplace_back("  %val = memref.load %viewed[%c1] : " + dst_type);
    lines.emplace_back("  %val_i8 = arith.trunci %val : i32 to i8");
    lines.emplace_back("  func.call @use(%val_i8) : (i8) -> ()");
  }
  else
  {
    lines.emplace_back("%c0xFFFFFFFF = arith.constant 4294967295 : i32");
    lines.emplace_back("memref.store %c0xFFFFFFFF, %viewed[%c1] : " + dst_type);
  }
//插入地址比较
  if (!distance.empty() && generate_preconditions_check_distance)
  {
    std::vector<std::string> precond1 = {
      "%c11 = arith.constant 11 : index", //size + 3 = 8 + 3 = 11
      "%too_far = arith.cmpi sgt, %" + distance + ", %c11 : index",
      "scf.if %too_far {",
      "  func.call @exit(%precond_fail) : (i32) -> ()",
     "  scf.yield",
     "}"
    };
    lines.insert(lines.begin(), precond1.begin(), precond1.end());
    auto precond2 = generate_preconditions_check_distance(distance);
    lines.insert(lines.begin(), precond2.begin(), precond2.end());
  }//TODO
  return lines;
}
