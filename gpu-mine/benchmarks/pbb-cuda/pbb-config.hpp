#pragma once
#include <algorithm>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fs = std::filesystem;
static constexpr const char* family_names[families] = {"qp","qd","pp","pd","cqd","cp","fp","wc","fs","pj"};
struct Options {
    std::string team = "shenjack", pattern = u8"\u6211\u4e5f\u4e0d\u77e5\u9053\u6211\u8981\u6d4b\u591a\u5c11 []";
    std::string backend, cuda_path = "split";
    std::optional<std::string> team_override, pattern_override;
    std::optional<Id> count, end;
    fs::path out, plan_file;
    U seconds = 60, batch = 0, block = 0, threads = 6, validate = 0, report = 10;
    Id start = 0;
    bool mine = false;
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
static Options arguments(int argc, const char* const* argv, std::string_view device) {
    Options o;
    bool timed = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view key = argv[i];
        if (key == "--mine") { o.mine = true; continue; }
        if (i + 1 == argc) throw std::runtime_error(u8"\u53c2\u6570\u7f3a\u5c11\u503c\uff1a" + std::string(key));
        const std::string_view val = argv[++i];
        if (key == "--out") o.out = fs::u8path(val);
        else if (key == "--plan") o.plan_file = fs::u8path(val);
        else if (key == "--backend") o.backend = val;
        else if (key == "--cuda-path") o.cuda_path = val;
        else if (key == "--seconds") { o.seconds = bounded(val, 1, 3600); timed = true; }
        else if (key == "--batch" || key == "--batch-size") o.batch = bounded(val, 1, 1048576);
        else if (key == "--block") o.block = bounded(val, 32, 256);
        else if (key == "--threads" || key == "--thread-count") o.threads = bounded(val, 1, 64);
        else if (key == "--validate") o.validate = bounded(val, 1, 10000000);
        else if (key == "--start") o.start = integer<Id>(val);
        else if (key == "--count") o.count = integer<Id>(val);
        else if (key == "--end") o.end = integer<Id>(val);
        else if (key == "--report-interval") o.report = bounded(val, 1, std::numeric_limits<U>::max());
        else if (key == "--team") o.team_override = std::string(val);
        else if (key == "--name-template" || key == "-m") o.pattern_override = std::string(val);
        else throw std::runtime_error(u8"\u672a\u77e5\u9009\u9879\uff1a" + std::string(key));
    }
    if (o.out.empty()) throw std::runtime_error(u8"\u8bf7\u901a\u8fc7 --out \u6307\u5b9a\u65b0\u76ee\u5f55\uff1b\u7528 --help \u67e5\u770b\u53c2\u6570");
    if (o.backend.empty()) o.backend = o.mine ? std::string(device) : "both";
    if (o.backend != device && o.backend != "cpu" && (o.mine || o.backend != "both"))
        throw std::runtime_error(u8"\u8be5\u4e8c\u8fdb\u5236\u7684 GPU \u540e\u7aef\u4e3a " + std::string(device));
    const bool tuned = o.mine && o.backend == device && device == "cuda" && o.cuda_path == "split";
    if (!o.batch) o.batch = tuned ? 1048576 : 262144;
    if (!o.block) o.block = tuned ? 128 : 64;
    const bool split_block = o.block == 32 || o.block == 64 || o.block == 128;
    if (o.block % 32 != 0 || (o.cuda_path == "split" && !split_block)
        || (o.cuda_path != "split" && (o.cuda_path != "direct" || device != "cuda")))
        throw std::runtime_error(u8"\u524d\u7b5b\u5de5\u4f5c\u7ec4\u652f\u6301 32\u300164\u3001128\uff1bCUDA direct \u53ef\u7528 32 \u7684\u500d\u6570\u81f3 256");
    if (o.mine) {
        if (timed || o.validate || (o.count && o.end))
            throw std::runtime_error(u8"--mine \u53ef\u9009 --count \u6216 --end\uff0c\u4e0e --seconds\u3001--validate \u4e92\u65a5");
        if (o.count && *o.count > std::numeric_limits<Id>::max() - o.start)
            throw std::runtime_error(u8"start + count \u8d85\u51fa u64 \u8303\u56f4");
        o.end = o.count ? o.start + *o.count : o.end.value_or(std::numeric_limits<Id>::max());
        if (*o.end < o.start) throw std::runtime_error(u8"end \u5fc5\u987b\u5927\u4e8e\u6216\u7b49\u4e8e start");
    } else if (o.end || o.count || o.start == std::numeric_limits<Id>::max()
               || (o.validate && (timed || o.validate > std::numeric_limits<Id>::max() - o.start))) {
        throw std::runtime_error(u8"\u57fa\u51c6\u4ec5\u63a5\u53d7 --seconds \u6216 --validate\uff0c\u4e14\u8d77\u59cb\u4f4d\u7f6e\u987b\u5bb9\u7eb3\u9a8c\u8bc1\u8303\u56f4");
    }
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
        bool first_line = true;
        while (std::getline(file, line)) {
            if (first_line && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
            first_line = false;
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
                    if (a != token.npos && m < 0)
                        throw std::runtime_error(u8"score_list \u4e2d\u7684\u672a\u77e5\u8bc4\u5206\u65cf " + std::string(token));
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
    if (opts.team_override) opts.team = *opts.team_override;
    if (opts.pattern_override) opts.pattern = *opts.pattern_override;
    if (!r.norder) for (int i = 0; i < families; ++i) { r.order[r.norder++] = i; r.enabled[i] = 1; }
    for (int i = 0; i < families; ++i) if (r.enabled[i] && (r.minimum[i] > 0 || r.potential[i] > 0)) r.any_gate = 1;
    return r;
}
static Launch prepare(const Options& opts, const Rules& plan) {
    const auto first = opts.pattern.find("[]");
    if (first == opts.pattern.npos || opts.pattern.find("[]", first + 2) != opts.pattern.npos
        || opts.pattern.size() + encoded_width - 2 > name_limit || opts.team.size() > 256
        || opts.pattern.find('\0') != opts.pattern.npos || opts.team.find('\0') != opts.team.npos)
        throw std::runtime_error(u8"PBB \u4ec5\u652f\u6301\u4e00\u4e2a []\uff0c\u751f\u6210\u7ed3\u679c\u6700\u591a 252 \u5b57\u8282\uff0c\u961f\u540d\u6700\u591a 256 \u5b57\u8282\uff0c\u5747\u4e0d\u542b NUL");
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
