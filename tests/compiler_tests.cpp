#include "compiler.hpp"
#include <iostream>
#include <stdexcept>
using namespace kforge;
void check(bool b) {
  if (!b)
    throw std::runtime_error("test failed");
}
void invalid(Graph g) {
  bool caught = false;
  try {
    compile(g);
  } catch (const std::invalid_argument &) {
    caught = true;
  }
  check(caught);
}
int main() {
  auto g = example(4);
  auto c = compile(g);
  check(c.live.size() == 5);
  check(c.unfused_bytes == 48 && c.fused_bytes == 16);
  check(c.source.find("v5") == std::string::npos);
  auto out = evaluate(g, {{-3, 2, 1, -2}, {1, 3, -4, -1}});
  check(out == std::vector<float>({0, 15, 12, 3}));
  auto bad = g;
  bad.nodes[1].type.shape = {3};
  invalid(bad);
  bad = g;
  bad.nodes[2].args = {2, 1};
  invalid(bad);
  bad = g;
  bad.nodes[5].args = {5, 5};
  invalid(bad);
  bad = g;
  bad.nodes[2].args = {99, 1};
  invalid(bad);
  bad = g;
  bad.nodes[0].type.shape = {0};
  invalid(bad);
  bad = g;
  bad.output = 999;
  invalid(bad);
  // Forward references are valid when the graph remains acyclic.
  Type t{DType::F32, {4}};
  Graph forward{{{Op::Relu, t, {1}}, {Op::Input, t, {}}}, 0};
  check(dependencies(forward) == std::vector<size_t>({1, 0}));
  std::cout << "DCE, fusion accounting, numerical reference, cycle, shape and DAG checks passed\n";
}
