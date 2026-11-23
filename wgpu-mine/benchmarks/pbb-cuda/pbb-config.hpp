#pragma once
#include <algorithm>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fs = std::filesystem;
static constexpr const char* family_names[families] = {"qp","qd","pp","pd","cqd","cp","fp","wc","fs","pj"};
struct Options {
    std::string team = "shenjack", pattern = "我也不知道我要测多少 []";
    std::string backend = "both", cuda_path = "split";
    fs::path out, plan_file;
    U seconds = 60, batch = 262144, block = 64, threads = 6, validate = 0;
    Id start = 0;
};

template<class T> static T integer(std::string_view s) {
    T n{};
    const auto [end, err] = std::from_chars(s.data(), s.data() + s.size(), n);
    if (err != std::errc{} || end != s.data() + s.size()) throw std::runtime_error("invalid integer: " + std::string(s));
    return n;
}
static U bounded(std::string_view s, U lo, U hi) {
    const auto n = integer<U>(s);
    if (n < lo || n > hi) throw std::runtime_error("integer out of range: " + std::string(s));
    return n;
}
static Options arguments(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::runtime_error("missing option value");
        std::string_view key = argv[i], val = argv[i+1];
        if (key == "--out") o.out = argv[i+1];
        else if (key == "--plan") o.plan_file = argv[i+1];
        else if (key == "--backend") o.backend = val;
        else if (key == "--cuda-path") o.cuda_path = val;
        else if (key == "--seconds") o.seconds = bounded(val, 1, 3600);
        else if (key == "--batch") o.batch = bounded(val, 1, 1048576);
        else if (key == "--block") o.block = bounded(val, 32, 256);
        else if (key == "--threads") o.threads = bounded(val, 1, 64);
        else if (key == "--validate") o.validate = bounded(val, 1, 10000000);
        else if (key == "--start") o.start = integer<Id>(val);
        else throw std::runtime_error("unknown option: " + std::string(key));
    }
    const bool split_block = o.block == 32 || o.block == 64 || o.block == 128;
    if (o.out.empty() || o.block % 32 != 0 || (o.cuda_path == "split" && !split_block) || o.start == std::numeric_limits<Id>::max()
        || (o.backend != "both" && o.backend != "cpu" && o.backend != "cuda")
        || (o.cuda_path != "split" && o.cuda_path != "direct"))
        throw std::runtime_error("use --out NEW_DIR [--plan FILE] [--backend both|cpu|cuda] [--cuda-path split|direct] [--seconds 60] [--validate N]");
    return o;
}
static int family(std::string_view key) {
    if (key == "rh") return 8;
    for (int i = 0; i < families; ++i) if (key == family_names[i]) return i;
    return -1;
}
static Rules rules(Options& opts) {
    Rules r{};
    if (opts.plan_file.empty()) {
        // 默认采用随包足球评分阈值；若有计划文件，缺省字段沿用原引擎的零值。
        const int minv[families] = {5400,6200,8000,8800,4000,4100,4200,4000,3900,4100};
        const int pttv[families] = {0,0,0,0,0,4500,4600,4400,4300,4500};
        std::copy(std::begin(minv), std::end(minv), r.minimum);
        std::copy(std::begin(pttv), std::end(pttv), r.potential);
        r.collect_mode = 1;
    } else {
        std::ifstream file(opts.plan_file, std::ios::binary);
        if (!file) throw std::runtime_error("cannot open plan");
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            const auto sep = line.find('=');
            if (sep == line.npos) throw std::runtime_error("missing = in plan");
            const std::string key = line.substr(0, sep), val = line.substr(sep + 1);
            const auto dec = [&]{ return integer<int>(val); };
            if (key == "team_name" || key == "team") opts.team = val;
            else if (key == "template") opts.pattern = val;
            else if (key == "score_mode") {
                if (val != "all" && val != "any") throw std::runtime_error("invalid score_mode");
                r.any = val == "any";
            } else if (key == "score_list") {
                r.norder = 0;
                std::fill(std::begin(r.enabled), std::end(r.enabled), 0);
                for (size_t begin = 0; begin < val.size();) {
                    const size_t at = val.find(',', begin);
                    const auto token = std::string_view(val).substr(begin, at == val.npos ? at : at - begin);
                    const auto a = token.find_first_not_of(" \t"), b = token.find_last_not_of(" \t");
                    const int m = a == token.npos ? -1 : family(token.substr(a, b - a + 1));
                    if (m >= 0 && !r.enabled[m]) { r.enabled[m] = 1; r.order[r.norder++] = m; }
                    if (at == val.npos) break;
                    begin = at + 1;
                }
            } else if (key.compare(0,4,"min_") == 0 && family(key.substr(4)) >= 0) r.minimum[family(key.substr(4))] = dec();
            else if (key.compare(0,4,"ptt_") == 0 && family(key.substr(4)) >= 5) r.potential[family(key.substr(4))] = dec();
            else if (key == "collect_mode") r.collect_mode = int(bounded(val, 0, 2));
            else if (key == "c_eight_v_min") r.eight = dec();
            else if (key == "c_seven_v_min") r.seven = dec();
            else if (key == "c_hp398_min") r.hp398 = dec();
            else if (key == "c_hl_min") r.floor = dec();
            else if (key == "output_xp") { (void)bounded(val, 0, 1); } // 此基准 CSV 始终保留全部评分字段。
            else throw std::runtime_error("unknown rule: " + key);
        }
        if (file.bad()) throw std::runtime_error("failed reading plan");
    }
    if (!r.norder) for (int i = 0; i < families; ++i) { r.order[r.norder++] = i; r.enabled[i] = 1; }
    for (int i = 0; i < families; ++i) if (r.enabled[i] && (r.minimum[i] > 0 || r.potential[i] > 0)) r.any_gate = 1;
    return r;
}
static Launch prepare(const Options& opts, const Rules& plan) {
    const auto first = opts.pattern.find("[]");
    if (first == opts.pattern.npos || opts.pattern.find("[]", first + 2) != opts.pattern.npos
        || opts.pattern.size() + encoded_width - 2 > name_limit || opts.team.size() > 256
        || opts.pattern.find('\0') != opts.pattern.npos || opts.team.find('\0') != opts.team.npos)
        throw std::runtime_error("PBB bench expects one [] and <=252 name bytes, no NUL");
    Launch input{};
    input.rules = plan;
    const auto suffix = opts.pattern.substr(first + 2);
    input.prefix_len = int(first);
    input.suffix_len = int(suffix.size());
    input.name_len = input.prefix_len + encoded_width + input.suffix_len;
    std::memcpy(input.prefix, opts.pattern.data(), first);
    std::memcpy(input.suffix, suffix.data(), suffix.size());
    char sample[name_limit + 1];
    make_name(input, 0, sample);
    Name seed{};
    seed.load_team(opts.team.c_str());
    seed.PRELEN = input.prefix_len + 1;
    seed.load_prefix(sample, input.name_len);
    std::memcpy(input.team_box, seed.val_base, 256);
    std::memcpy(input.prefix_box, seed.val_base2, 256);
    input.i_pre = seed.i_pre; input.j_pre = seed.j_pre; input.s_pre = seed.s_pre;
    return input;
}
