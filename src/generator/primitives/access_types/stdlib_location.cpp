/*
 * This file is distributed under the Apache License, Version 2.0; refer to
 * LICENSE for details.
 *
 * Initial author: Emanuel Vintila
 */

#include "stdlib_location.h"

#include <cassert>
#include <atomic>
#include <string>

#include "misc.h"
static std::string get_unique_read_value_name()
{
  static std::atomic<size_t> counter{0};
  return "read_value_" + std::to_string(counter.fetch_add(1));
}
#include "generator/primitives/access_types/read_action.h"

std::vector<std::string> StdlibLocation::generate(
  std::shared_ptr<AccessAction> action, const std::string &access_var_name, size_t size, size_t array_size, const std::string &index_var, bool needs_strided, const std::string &offset, const std::string &custom_type) const
{
  return generate_split_const_vars(action, access_var_name, size, array_size, index_var, needs_strided, offset, custom_type).to_lines();
}

AccessLocation::SplitAccess StdlibLocation::generate_split_aux_vars(
  std::shared_ptr<AccessAction> action, const std::string &access_var_name, size_t size, size_t array_size, const std::string &index_var, std::function<std::vector<std::string>(const std::string&)> generate_counter_update,
  const std::string &distance, bool needs_strided, const std::string &offset) const
{
  SplitAccess split_access;
  std::string size_str = std::to_string(size);
  if (is_a<ReadAction>(action))
  {
    // READ: memref.copy
    if (array_size > 0) {
      std::string array_size_str = std::to_string(array_size);
      std::string rv = get_unique_read_value_name();
      std::string strided_type = "memref<1x" + size_str + "xi8, strided<[" + size_str + ", 1], offset: ?>>";
      split_access.aux_variables = {
        {"c0", "index", "", "0"},
        {rv, "memref<1x" + size_str + "xi8>", "", "memref.alloca() : memref<1x" + size_str + "xi8>"},
      };
      split_access.access_lines = {
        "%subview_tmp = memref.subview %" + access_var_name + "[%" + index_var + ", 0][1, " + size_str + "][1, 1] : memref<" + array_size_str + "x" + size_str + "xi8> to " + strided_type,
        "memref.copy %subview_tmp, %" + rv + " : " + strided_type + " to memref<1x" + size_str + "xi8>"
      };
      split_access.access_lines.emplace_back("%use_val_" + rv + " = memref.load %" + rv + "[%c0, %c0] : memref<1x" + size_str + "xi8>");
      split_access.access_lines.emplace_back("func.call @use(%use_val_" + rv + ") : (i8) -> ()");
    } else {
      std::string rv = get_unique_read_value_name();
      split_access.aux_variables = {
        {"c0", "index", "", "0"},
        {rv, "memref<" + size_str + "xi8>", "", "memref.alloca() : memref<" + size_str + "xi8>"},
      };
      std::string src_type = needs_strided ? ("memref<" + size_str + "xi8, strided<[1], offset: "+ offset +">>") : ("memref<" + size_str + "xi8>");
      split_access.access_lines.push_back( "memref.copy %" + access_var_name + ", %" + rv + " : " + src_type + " to memref<" + size_str + "xi8>" );
      split_access.access_lines.emplace_back("%use_val_" + rv + " = memref.load %" + rv + "[%c0] : memref<" + size_str + "xi8>");
      split_access.access_lines.emplace_back("func.call @use(%use_val_" + rv + ") : (i8) -> ()");
    }
    split_access.result = access_var_name;
  }
  else
  {
    // WRITE: scf.for + store
    split_access.aux_variables = {
      {"c0", "index", "", "0"},
      {"c1", "index", "", "1"},
      {"c" + size_str, "index", "", size_str},
      {"c0xFF", "i8", "", "255"},
    };
    if (array_size > 0) {
      std::string array_size_str = std::to_string(array_size);
      std::string type_suffix = needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">";
      split_access.access_lines.emplace_back("scf.for %i = %c0 to %c" + size_str + " step %c1 {");
      split_access.access_lines.emplace_back("  memref.store %c0xFF, %" + access_var_name + "[%" + index_var + ", %i] : memref<" + array_size_str + "x" + size_str + "xi8" + type_suffix);
      split_access.access_lines.emplace_back("}");
    } else {
      std::string type_suffix = needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">";
      split_access.access_lines.emplace_back("scf.for %i = %c0 to %c" + size_str + " step %c1 {");
      split_access.access_lines.emplace_back("  memref.store %c0xFF, %" + access_var_name + "[%i] : memref<" + size_str + "xi8" + type_suffix);
      split_access.access_lines.emplace_back("}");
    }
    split_access.result = access_var_name;
  }
  split_access.description = "auxiliary variables";
  return split_access;
}

AccessLocation::SplitAccess StdlibLocation::generate_split_const_vars(
  std::shared_ptr<AccessAction> action, const std::string &access_var_name, size_t size, size_t array_size, const std::string &index_var, bool needs_strided, const std::string &offset, const std::string &custom_type) const
{
  SplitAccess split_access;
  std::string size_str = std::to_string(size);
  std::string type_suffix = needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">";
  std::string base_type;
  if (!custom_type.empty()) {
    base_type = custom_type;
  } else {
    base_type = "memref<" + size_str  + (array_size > 0 ? "x" + std::to_string(array_size) : "") + "xi8" + type_suffix;
  }
  if (is_a<ReadAction>(action))
  {
    // READ: memref.copy
    if (array_size > 0) {
      std::string array_size_str = std::to_string(array_size);
      std::string rv = get_unique_read_value_name();
      std::string strided_type = "memref<1x" + size_str + "xi8, strided<[" + size_str + ", 1], offset: ?>>";
      split_access.aux_variables = {
        {"c0", "index", "", "0"},
        {rv, "memref<1x" + size_str + "xi8>", "", "memref.alloca() : memref<1x" + size_str + "xi8>"},
      };
      split_access.access_lines = {
        "%subview_tmp = memref.subview %" + access_var_name + "[%" + index_var + ", 0][1, " + size_str + "][1, 1] :" + base_type +" to " + strided_type,
        "memref.copy %subview_tmp, %" + rv + " : " + strided_type + " to memref<1x" + size_str + "xi8>"
      };
      split_access.access_lines.emplace_back("%use_val_" + rv + " = memref.load %" + rv + "[%c0, %c0] : memref<1x" + size_str + "xi8>");
      split_access.access_lines.emplace_back("func.call @use(%use_val_" + rv + ") : (i8) -> ()");
    } else {
      std::string rv = get_unique_read_value_name();
      split_access.aux_variables = {
        {"c0", "index", "", "0"},
        {rv, "memref<" + size_str + "xi8>", "", "memref.alloca() : memref<" + size_str + "xi8>"},
      };
      std::string src_type = needs_strided ? ("memref<" + size_str + "xi8, strided<[1], offset: " + offset + ">>") : base_type;
      split_access.access_lines.push_back( "memref.copy %" + access_var_name + ", %" + rv + " : " + src_type + " to memref<" + size_str + "xi8>" );
      split_access.access_lines.emplace_back("%use_val_" + rv + " = memref.load %" + rv + "[%c0] : memref<" + size_str + "xi8>");
      split_access.access_lines.emplace_back("func.call @use(%use_val_" + rv + ") : (i8) -> ()");
    }
    split_access.result = access_var_name;
  }
  else
  {
    // WRITE: scf.for + store
    split_access.aux_variables = {
      {"c0", "index", "", "0"},
      {"c1", "index", "", "1"},
      {"c" + size_str, "index", "", size_str},
      {"c0xFF", "i8", "", "255"},
    };
    if (array_size > 0) {
      std::string array_size_str = std::to_string(array_size);
      std::string type_suffix = needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">";
      split_access.access_lines.emplace_back("scf.for %i = %c0 to %c" + size_str + " step %c1 {");
        split_access.access_lines.emplace_back("  memref.store %c0xFF, %" + access_var_name + "[%" + index_var + ", %i] : memref<" + array_size_str + "x" + size_str + "xi8" + type_suffix);
      split_access.access_lines.emplace_back("}");
    } else {
      split_access.access_lines.emplace_back("scf.for %i = %c0 to %c" + size_str + " step %c1 {");
        split_access.access_lines.emplace_back("  memref.store %c0xFF, %" + access_var_name + "[%i] : " + base_type);
      split_access.access_lines.emplace_back("}");
    }
    split_access.result = access_var_name;
  }
  split_access.description = "constants";
  return split_access;
}

AccessLocation::SplitAccess StdlibLocation::generate_bulk_split_using_index(
  std::shared_ptr<AccessAction> action,
  std::string from,
  std::string to,
  std::string distance,
  std::function<std::vector<std::string>(const std::string&)> generate_preconditions_check_distance,
  std::function<std::vector<std::string>(const std::string&, const std::string&, const std::string&)> generate_preconditions_check_in_range,
  std::function<std::vector<std::string>(const std::string&)> generate_counter_update,
  bool needs_strided,
  const std::string &offset
) const
{
  SplitAccess split_access;
  if ( distance.empty() )
  {
    distance = "distance"; // placeholder; caller provides actual distance SSA
  }
  std::string dist = (is_number(distance) && std::stoll(distance) == 0) ? "c" + distance : distance;
  std::string negadist_val;
  if(distance.find("negated") != std::string::npos && needs_strided) {
    dist = "negadist_variant";
    negadist_val = "%" + dist + " = arith.subi %c0, %" + distance + " : index";
  }

  if (is_a<ReadAction>(action))
  {
    // READ: dynamic chunk copy using memref.copy + subview
    std::string rv = get_unique_read_value_name();
    split_access.aux_variables = {
      {"c0", "index", "", "0"},
      {"c1", "index", "", "1"},
      {"c1024", "index", "", "1024"},
      {rv, "memref<1024xi8>", "", "memref.alloca() : memref<1024xi8>"},
    };
    split_access.access_lines = {
      negadist_val,
      "scf.for %i = %c0 to %" + dist + " step %c1024 {",
      "  %remaining = arith.subi %" + dist + ", %i : index",
      "  %is_full = arith.cmpi sgt, %remaining, %c1024 : index",
      "  %step = arith.select %is_full, %c1024, %remaining : index",
      "  %src_slice = memref.subview %" + from + "[%i][%step][1] : memref<8xi8" + (needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">") + " to memref<?xi8, strided<[1], offset: ?>>",
      "  %dst_slice = memref.subview %" + rv + "[%c0][%step][1] : memref<1024xi8> to memref<?xi8, strided<[1], offset: ?>>",
      "  memref.copy %src_slice, %dst_slice : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>",
      "}",
    };
    if (generate_preconditions_check_distance)
    {
      auto preconds = generate_preconditions_check_distance(distance);
      split_access.access_lines.insert(split_access.access_lines.begin(), preconds.begin(), preconds.end());
    }
    split_access.access_lines.emplace_back("%use_val_" + rv + " = memref.load %" + rv + "[%c0] : memref<1024xi8>");
    split_access.access_lines.emplace_back("func.call @use(%use_val_" + rv + ") : (i8) -> ()");
    split_access.result = from;
  }
  else
  {
    // WRITE: scf.for + store per byte
    std::string type_suffix = needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">";
    split_access.aux_variables = {
      {"c0", "index", "", "0"},
      {"c1", "index", "", "1"},
      {"c0xFF", "i8", "", "255"},
    };
    // if (generate_preconditions_check_in_range)
    // {
    //   split_access.access_lines.insert( split_access.access_lines.end(), {"if ( " + generate_preconditions_check_in_range("i", from, to) + " ) _exit(PRECONDITIONS_FAILED_VALUE);"});
    //   split_access.access_lines.insert( split_access.access_lines.end(), {"if ( " + generate_preconditions_check_in_range("step_distance", from, to) + " ) _exit(PRECONDITIONS_FAILED_VALUE);"});
    // }//TODO
    split_access.access_lines = {
      negadist_val,
      "scf.for %i = %c0 to %" + dist + " step %c1 {",
      "  memref.store %c0xFF, %" + from + "[%i] : memref<8xi8" + type_suffix,
      "}",
    };
    if (generate_preconditions_check_distance)
    {
      auto preconds = generate_preconditions_check_distance(distance);
      split_access.access_lines.insert(split_access.access_lines.begin(), preconds.begin(), preconds.end());
    }
    split_access.result = from;
  }
  split_access.description = "index";
  return split_access;
}
AccessLocation::SplitAccess StdlibLocation::generate_bulk_split_using_aux_ptr(
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
  SplitAccess split_access;
  std::string dist = (is_number(distance) && std::stoll(distance) == 0) ? "c" + distance : distance;
  bool is_underflow = false;
  std::vector<std::string> counter_update = generate_counter_update("reach_index");
  for (const auto& line : counter_update)
  {
    if (line.find("subi") != std::string::npos)
    {
      is_underflow = true;
      break;
    }
  }

  if (is_a<ReadAction>(action))
  {
    // READ: dynamic chunk copy using memref.copy + subview
    std::string rv = get_unique_read_value_name();
    split_access.aux_variables = {
      {"c0", "index", "", "0"},
      {"c1", "index", "", "1"},
      {"c1024", "index", "", "1024"},
      {rv, "memref<1024xi8>", "", "memref.alloca() : memref<1024xi8>"},
    };
    split_access.access_lines = {
      "scf.for %i = %c0 to %" + dist + " step %c1024 {",
      "  %remaining = arith.subi %" + dist + ", %i : index",
      "  %is_full = arith.cmpi sgt, %remaining, %c1024 : index",
      "  %step = arith.select %is_full, %c1024, %remaining : index",
      "  %src_slice = memref.subview %" + from + "[%i][%step][1] : memref<8xi8" + (needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">") + " to memref<?xi8, strided<[1], offset: ?>>",
      "  %dst_slice = memref.subview %" + rv + "[%c0][%step][1] : memref<1024xi8> to memref<?xi8, strided<[1], offset: ?>>",
      "  memref.copy %src_slice, %dst_slice : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>",
      "}",
    };
    if (generate_preconditions_check_distance)
    {
      auto preconds = generate_preconditions_check_distance(distance);
      split_access.access_lines.insert(split_access.access_lines.begin(), preconds.begin(), preconds.end());
    }
    split_access.access_lines.emplace_back("%use_val_" + rv + " = memref.load %" + rv + "[%c0] : memref<1024xi8>");
    split_access.access_lines.emplace_back("func.call @use(%use_val_" + rv + ") : (i8) -> ()");
    split_access.result = from;
  }
  else
  {
    // WRITE: scf.for + store per byte
    std::string type_suffix = needs_strided ? ", strided<[1], offset: " + offset + ">>" : ">";
    split_access.aux_variables = {
      {"c0", "index", "", "0"},
      {"c1", "index", "", "1"},
      {"c0xFF", "i8", "", "255"},
    };
    split_access.access_lines = {};
    // if (generate_preconditions_check_in_range)
    // {
    //   split_access.access_lines.push_back( "if ( " + generate_preconditions_check_in_range("aux_ptr", from, to) + " ) _exit(PRECONDITIONS_FAILED_VALUE);" );
    //   split_access.access_lines.push_back( "if ( " + generate_preconditions_check_in_range("step_distance", from, to) + " ) _exit(PRECONDITIONS_FAILED_VALUE);" );
    // }

    split_access.access_lines = {
      "scf.for %i = %c0 to %" + dist + " step %c1 {",
      "  memref.store %c0xFF, %" + from + "[%i] : memref<8xi8" + type_suffix,
      "}",
    };
    if (generate_preconditions_check_distance)
    {
      auto preconds = generate_preconditions_check_distance(distance);
      split_access.access_lines.insert(split_access.access_lines.begin(), preconds.begin(), preconds.end());
    }
    split_access.result = from;
  }

  split_access.description = "auxiliary pointer";
  return split_access;
}


std::vector<std::string> StdlibLocation::generate_at_index(
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
  std::string src_type = needs_strided ? "memref<" + size_str + "xi8, strided<[1], offset: " + offset + ">>" : "memref<" + size_str + "xi8>";
  std::string dst_type = "memref<" + size_str + "xi8, strided<[1], offset: ?>>";
  // std::string dst_type = needs_strided ? "memref<" + size_str + "xi8, strided<[1], offset: ?>>" : "memref<" + size_str + "xi8>";
  if (is_a<ReadAction>(action))
  {
    // READ: memref.copy via subview
    std::string rv = get_unique_read_value_name();
    lines = {
      "%c0 = arith.constant 0 : index",
      "%" + rv + " = memref.alloca() : memref<" + size_str + "xi8>",
      "%src_slice = memref.subview %" + access_var_name + "[%" + index + "][" + size_str + "][1] : " + src_type + " to " + dst_type,
      "memref.copy %src_slice, %" + rv + " : " + dst_type + " to memref<" + size_str + "xi8>",
      "%use_val_" + rv + " = memref.load %" + rv + "[%c0] : memref<" + size_str + "xi8>",
      "func.call @use(%use_val_" + rv + ") : (i8) -> ()",
    };
    if (generate_preconditions_check_distance)
    {
      auto preconds = generate_preconditions_check_distance(index);
      lines.insert(lines.begin(), preconds.begin(), preconds.end());
    }
  }
  else
  {
    // WRITE: scf.for + store
    lines = {
      "%c0 = arith.constant 0 : index",
      "%c1 = arith.constant 1 : index",
      "%c" + size_str + " = arith.constant " + size_str + " : index",
      "%c0xFF = arith.constant 255 : i8",
      "scf.for %j = %c0 to %c" + size_str + " step %c1 {",
      "  %idx = arith.addi %j, %" + index + " : index",
      "  memref.store %c0xFF, %" + access_var_name + "[%idx] : " + src_type,
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


std::vector<std::string> StdlibLocation::generate_using_runtime_index(
  std::shared_ptr<AccessAction> action,
  const std::string &access_var_name,
  std::string index,
  std::string distance,
  std::function<std::vector<std::string>(const std::string&)> generate_preconditions_check_distance,
  std::function<std::vector<std::string>(const std::string&, const std::string&, const std::string&)> generate_preconditions_check_in_range
) const
{
  assert(0 && "generate_using_runtime_index not used for StdlibLocation");
  return {};
}

std::vector<std::string> StdlibLocation::generate_uint32(
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
  assert(0 && "generate_uint32 not used for StdlibLocation");
  return {};
}

std::vector<std::string> StdlibLocation::generate_uint8(
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
  assert(0 && "generate_uint8 not used for StdlibLocation");
  return {};
}

std::vector<std::string> StdlibLocation::generate_big_type(
  std::shared_ptr<AccessAction> action,
  const std::string &orig_var_name,
  const std::string &orig_type,
  const std::string &view_offset,
  const std::string &view_sizes,
  const std::string &distance
) const
{
  assert(0 && "generate_big_type not used for StdlibLocation");
  return {};
}

std::vector<std::string> StdlibLocation::generate_load_widening(
  std::shared_ptr<AccessAction> action,
  const std::string &orig_var_name,
  const std::string &orig_type,
  const std::string &view_offset,
  const std::string &access_index
) const
{
  assert(0 && "generate_load_widening not used for StdlibLocation");
  return {};
}
