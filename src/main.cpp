#include "compiler.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
int main(int argc, char **argv) {
  try {
    auto g = kforge::example();
    if (argc == 2 && std::string(argv[1]) == "--benchmark") {
      for (int i = 0; i < 100; ++i)
        (void)kforge::compile(g);
      std::vector<double> samples;
      size_t bytes = 0;
      for (int i = 0; i < 10000; ++i) {
        auto a = std::chrono::steady_clock::now();
        auto c = kforge::compile(g);
        auto b = std::chrono::steady_clock::now();
        bytes += c.source.size();
        samples.push_back(std::chrono::duration<double, std::micro>(b - a).count());
      }
      std::sort(samples.begin(), samples.end());
      auto c = kforge::compile(g);
      std::cout << "{\"iterations\":10000,\"p50_us\":" << samples[5000]
                << ",\"p95_us\":" << samples[9500]
                << ",\"unfused_materialization_bytes\":" << c.unfused_bytes
                << ",\"fused_materialization_bytes\":" << c.fused_bytes
                << ",\"source_bytes_checksum\":" << bytes << "}\n";
    } else if (argc == 1)
      std::cout << kforge::compile(g).source;
    else
      throw std::invalid_argument("usage: kforge [--benchmark]");
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
