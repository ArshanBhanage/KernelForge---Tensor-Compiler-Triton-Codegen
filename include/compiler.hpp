#pragma once
#include <cstddef>
#include <string>
#include <vector>
namespace kforge {
enum class Op { Input, Add, Mul, Relu };
enum class DType { F32 };
struct Type {
  DType dtype = DType::F32;
  std::vector<size_t> shape;
  bool operator==(const Type &) const = default;
};
struct Node {
  Op op;
  Type type;
  std::vector<size_t> args;
};
struct Graph {
  std::vector<Node> nodes;
  size_t output;
};
struct Compiled {
  std::string source;
  std::vector<size_t> live;
  size_t unfused_bytes, fused_bytes;
};
size_t elements(const Type &);
std::vector<size_t> dependencies(const Graph &);
Compiled compile(const Graph &);
std::vector<float> evaluate(const Graph &, const std::vector<std::vector<float>> &inputs);
Graph example(size_t n = 1024);
} // namespace kforge
