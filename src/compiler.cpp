#include "compiler.hpp"
#include <algorithm>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
namespace kforge {
size_t elements(const Type &t) {
  if (t.dtype != DType::F32 || t.shape.empty())
    throw std::invalid_argument("expected ranked FP32 tensor");
  size_t n = 1;
  for (size_t d : t.shape) {
    if (!d || n > std::numeric_limits<size_t>::max() / d)
      throw std::invalid_argument("invalid shape");
    n *= d;
  }
  return n;
}
std::vector<size_t> dependencies(const Graph &g) {
  if (g.output >= g.nodes.size())
    throw std::invalid_argument("invalid output");
  // Validate even dead nodes: malformed IR must not silently disappear in DCE.
  for (auto &n : g.nodes) {
    elements(n.type);
    size_t arity;
    switch (n.op) {
    case Op::Input:
      arity = 0;
      break;
    case Op::Relu:
      arity = 1;
      break;
    case Op::Add:
    case Op::Mul:
      arity = 2;
      break;
    default:
      throw std::invalid_argument("unknown operation");
    }
    if (n.args.size() != arity)
      throw std::invalid_argument("invalid arity");
    for (auto a : n.args)
      if (a >= g.nodes.size() || !(g.nodes[a].type == n.type))
        throw std::invalid_argument("operand type or shape mismatch");
  }
  std::vector<int> state(g.nodes.size());
  std::vector<size_t> order;
  std::function<void(size_t)> visit = [&](size_t i) {
    if (state[i] == 1)
      throw std::invalid_argument("dependency cycle");
    if (state[i] == 2)
      return;
    state[i] = 1;
    for (auto a : g.nodes[i].args)
      visit(a);
    state[i] = 2;
    order.push_back(i);
  };
  for (size_t i = 0; i < g.nodes.size(); ++i)
    visit(i);
  std::fill(state.begin(), state.end(), 0);
  order.clear();
  visit(g.output);
  return order;
}
Compiled compile(const Graph &g) {
  auto order = dependencies(g);
  std::ostringstream s;
  s << "import triton\nimport triton.language as tl\n\n@triton.jit\ndef kernel(";
  for (auto i : order)
    if (g.nodes[i].op == Op::Input)
      s << "p" << i << ", ";
  s << "out, N: tl.constexpr, BLOCK: tl.constexpr):\n    x = tl.program_id(0) * BLOCK + "
       "tl.arange(0, BLOCK)\n    mask = x < N\n";
  size_t materialized = 0;
  for (auto i : order) {
    auto &n = g.nodes[i];
    s << "    v" << i << " = ";
    switch (n.op) {
    case Op::Input:
      s << "tl.load(p" << i << " + x, mask=mask, other=0.0)";
      break;
    case Op::Add:
    case Op::Mul:
      s << "v" << n.args[0] << (n.op == Op::Add ? " + " : " * ") << "v" << n.args[1];
      ++materialized;
      break;
    case Op::Relu:
      s << "tl.maximum(v" << n.args[0] << ", 0.0)";
      ++materialized;
      break;
    }
    s << "\n";
  }
  s << "    tl.store(out + x, v" << g.output << ", mask=mask)\n";
  size_t n = elements(g.nodes[g.output].type);
  if (n > std::numeric_limits<size_t>::max() / 4 / std::max(size_t(1), materialized))
    throw std::invalid_argument("byte count overflow");
  return {s.str(), order, materialized * n * 4, n * 4};
}
std::vector<float> evaluate(const Graph &g, const std::vector<std::vector<float>> &inputs) {
  auto order = dependencies(g);
  std::vector<std::vector<float>> values(g.nodes.size());
  for (auto i : order) {
    auto &n = g.nodes[i];
    size_t count = elements(n.type);
    auto &v = values[i];
    v.resize(count);
    if (n.op == Op::Input) {
      if (i >= inputs.size() || inputs[i].size() != count)
        throw std::invalid_argument("input size mismatch");
      v = inputs[i];
      continue;
    }
    for (size_t x = 0; x < count; ++x) {
      float a = values[n.args[0]][x];
      if (n.op == Op::Relu)
        v[x] = std::max(a, 0.f);
      else {
        float b = values[n.args[1]][x];
        v[x] = n.op == Op::Add ? a + b : a * b;
      }
    }
  }
  return values[g.output];
}
Graph example(size_t n) {
  Type t{DType::F32, {n}};
  return {{{Op::Input, t, {}},
           {Op::Input, t, {}},
           {Op::Add, t, {0, 1}},
           {Op::Mul, t, {2, 1}},
           {Op::Relu, t, {3}},
           {Op::Mul, t, {0, 0}}},
          4};
}
} // namespace kforge
