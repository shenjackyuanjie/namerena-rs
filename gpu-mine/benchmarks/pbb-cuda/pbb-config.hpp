#pragma once
#include <algorithm>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
static constexpr const char* family_names[families] = {"qp","qd","pp","pd","cqd","cp","fp","wc","fs","pj"};
struct Options {
    std::string team = "shenjack", pattern = u8"\u6211\u4e5f\u4e0d\u77e5\u9053\u6211\u8981\u6d4b\u591a\u5c11 []";
    std::string backend, cuda_path = "split";
    std::optional<std::string> team_override, pattern_override;
    std::optional<Id> count, end, encode, seed;
    std::optional<std::string> decode;
    fs::path fill_plan;
    U shuffle = 0;
    fs::path out, plan_file;
    U seconds = 60, batch = 0, block = 0, threads = 6, validate = 0, report = 10;
    Id start = 0;
    bool mine = false, cpu_scalar = false;
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
    const bool cpu_only = device == "cpu";
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
        else if (key == "--cpu-path" && cpu_only) {
            if (val != "scalar" && val != "fast") throw std::runtime_error("--cpu-path: fast | scalar");
            o.cpu_scalar = val == "scalar";
        }
        else if (key == "--seconds") { o.seconds = bounded(val, 1, 3600); timed = true; }
        else if (key == "--batch" || key == "--batch-size") o.batch = bounded(val, 1, 16777216);
        else if (key == "--block") o.block = bounded(val, 32, 256);
        else if (key == "--threads" || key == "--thread-count") o.threads = bounded(val, 1, 64);
        else if (key == "--validate") o.validate = bounded(val, 1, 10000000);
        else if (key == "--start") o.start = integer<Id>(val);
        else if (key == "--count") o.count = integer<Id>(val);
        else if (key == "--end") o.end = integer<Id>(val);
        else if (key == "--report-interval") o.report = bounded(val, 1, std::numeric_limits<U>::max());
        else if (key == "--team") o.team_override = std::string(val);
        else if (key == "--name-template" || key == "-m") o.pattern_override = std::string(val);
        else if (key == "--fill-plan") o.fill_plan = fs::u8path(val);
        else if (key == "--encode-id") o.encode = integer<Id>(val);
        else if (key == "--decode-fill") o.decode = std::string(val);
        else if (key == "--seed") o.seed = integer<Id>(val);
        else if (key == "--shuffle") {
            if (val == "mix") o.shuffle = 1;
            else if (val == "affine") o.shuffle = 2;
            else if (val == "feistel") o.shuffle = 3;
            else throw std::runtime_error(u8"--shuffle \u53ef\u9009 affine\u3001mix\u3001feistel");
        }
        else throw std::runtime_error(u8"\u672a\u77e5\u9009\u9879\uff1a" + std::string(key));
    }
    if (o.seed && !o.shuffle) throw std::runtime_error(u8"--seed \u9700\u8981\u914d\u5408 --shuffle");
    if (o.encode || o.decode) {
        if ((o.encode && o.decode) || o.mine || timed || o.validate || o.count || o.end || o.start)
            throw std::runtime_error(u8"\u7f16\u7801\u6216\u89e3\u7801\u4e0d\u80fd\u4e0e\u6d4b\u53f7\u6216\u8303\u56f4\u9009\u9879\u540c\u7528");
        return o;
    }
    if (o.out.empty()) throw std::runtime_error(u8"\u8bf7\u901a\u8fc7 --out \u6307\u5b9a\u65b0\u76ee\u5f55\uff1b\u7528 --help \u67e5\u770b\u53c2\u6570");
    if (o.backend.empty()) o.backend = (o.mine || cpu_only) ? std::string(device) : "both";
    if (cpu_only && (o.backend != "cpu" || o.cuda_path != "split"))
        throw std::runtime_error("tswn pbb: --backend cpu");
    if (o.validate && o.cpu_scalar)
        throw std::runtime_error("--validate: --cpu-path fast");
    if (o.backend != device && o.backend != "cpu" && (o.mine || o.backend != "both"))
        throw std::runtime_error(u8"\u8be5\u4e8c\u8fdb\u5236\u7684 GPU \u540e\u7aef\u4e3a " + std::string(device));
    const bool tuned = o.mine && o.backend == device && device == "cuda" && o.cuda_path == "split";
    if (!o.batch) o.batch = tuned ? 1048576 : 262144;
    if (!o.block) o.block = tuned ? 128 : 64;
    const bool split_block = o.block == 32 || o.block == 64 || o.block == 128
        || ((device == "cuda" || cpu_only) && o.block == 96);
    if (o.block % 32 != 0 || (o.cuda_path == "split" && !split_block)
        || (o.cuda_path != "split" && (o.cuda_path != "direct" || device != "cuda")))
        throw std::runtime_error(u8"WGSL \u5de5\u4f5c\u7ec4\u652f\u6301 32/64/128\uff1bCUDA split \u652f\u6301 32/64/96/128\uff1bCUDA direct \u652f\u6301 32 \u7684\u500d\u6570\u81f3 256");
    if (o.mine) {
        if (timed || o.validate || (o.count && o.end))
            throw std::runtime_error(u8"--mine \u53ef\u9009 --count \u6216 --end\uff0c\u4e0e --seconds\u3001--validate \u4e92\u65a5");
        if (o.count && *o.count > std::numeric_limits<Id>::max() - o.start)
            throw std::runtime_error(u8"start + count \u8d85\u51fa u64 \u8303\u56f4");
        if (o.count) o.end = o.start + *o.count;
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
struct FillPlan {
    std::vector<FillSlot> slots;
    std::vector<U> data;
    Id domain = 0; // 完整 u64 不需要有限区间上界。
    int length = encoded_width;
};

static bool config_line(std::istream& in, std::string& s, bool& first) {
    if (!std::getline(in, s)) {
        if (in.bad()) throw std::runtime_error(u8"\u8bfb\u53d6\u586b\u5145\u6587\u4ef6\u5931\u8d25");
        return false;
    }
    if (first && s.compare(0, 3, "\xef\xbb\xbf") == 0) s.erase(0, 3);
    first = false;
    if (!s.empty() && s.back() == '\r') s.pop_back();
    return true;
}

// 仅接受标准 UTF-8 标量；拒绝截断、过长编码、代理项和 NUL。
static U read_scalar(std::string_view text, size_t& pos) {
    const auto byte = [&](size_t i) { return static_cast<unsigned char>(text[i]); };
    const U b = byte(pos++);
    if (b > 0 && b < 0x80) return b;
    const U width = b >= 0xc2 && b <= 0xdf ? 2 : b >= 0xe0 && b <= 0xef ? 3 : b >= 0xf0 && b <= 0xf4 ? 4 : 0;
    if (width && text.size() - pos >= width - 1) {
        U c = b & (0x7fu >> width);
        for (U i = 1; i < width; ++i) {
            const U next = byte(pos++);
            if ((next & 0xc0u) != 0x80u) throw std::runtime_error(u8"\u586b\u5145\u542b\u65e0\u6548 UTF-8");
            c = (c << 6) | (next & 63u);
        }
        if (c >= (width == 2 ? 0x80u : width == 3 ? 0x800u : 0x10000u) && c <= 0x10ffffu
            && (c < 0xd800u || c > 0xdfffu)) return c;
    }
    throw std::runtime_error(u8"\u586b\u5145\u542b\u65e0\u6548 UTF-8 \u6216 NUL");
}

static bool fill_family(std::string_view name) {
    return !name.empty() && name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-") == name.npos;
}

static FillPlan load_fill(const fs::path& path) {
    FillPlan plan;
    if (path.empty()) return plan;
    std::ifstream cfg(path, std::ios::binary);
    if (!cfg) throw std::runtime_error(u8"\u65e0\u6cd5\u6253\u5f00\u586b\u5145\u8ba1\u5212");
    struct Source { bool file; std::string value; };
    std::map<std::string, Source> sources;
    std::optional<std::string> format;
    std::string line;
    bool first = true;
    while (config_line(cfg, line, first)) {
        if (line.empty() || line[0] == '#') continue;
        const auto at = line.find('=');
        if (at == line.npos) throw std::runtime_error(u8"\u586b\u5145\u8ba1\u5212\u6bcf\u9879\u9700\u8981 key=value");
        const auto key = line.substr(0, at), value = line.substr(at + 1);
        if (key == "format" && !format) { format = value; continue; }
        const bool file = key.compare(0, 5, "file.") == 0;
        const auto prefix = file ? 5u : 6u;
        if ((!file && key.compare(0, 6, "chars.") != 0) || !fill_family(std::string_view(key).substr(prefix))
            || !sources.emplace(key.substr(prefix), Source{file, value}).second)
            throw std::runtime_error(u8"\u672a\u77e5\u6216\u91cd\u590d\u7684\u586b\u5145\u8ba1\u5212\u952e\uff1a" + key);
    }
    if (!format || format->size() > 4096) throw std::runtime_error(u8"\u586b\u5145\u8ba1\u5212\u9700\u8981\u6709\u6548\u7684 format");
    constexpr size_t table_limit = 1u << 22; // u32 每字节，显存最多 16 MiB。
    std::vector<U> tables;
    std::map<std::string, FillSlot> loaded;
    auto get_slot = [&](const std::string& name) {
        const auto prev = loaded.find(name);
        if (prev != loaded.end()) return prev->second;
        const auto src = sources.find(name);
        FillSlot slot{0, 0, 0, U(tables.size())};
        std::unordered_set<std::string> seen;
        const auto append = [&](std::string_view entry) {
            if (entry.empty() || entry.size() > name_limit || (slot.width && entry.size() != slot.width))
                throw std::runtime_error(u8"\u586b\u5145\u65cf\u6761\u76ee\u5fc5\u987b\u7b49\u5bbd\u4e14\u975e\u7a7a\uff1a" + name);
            if (slot.radix == 65536 || tables.size() + entry.size() > table_limit - name_limit)
                throw std::runtime_error(u8"\u586b\u5145\u65cf\u6216\u8bcd\u8868\u8d85\u8fc7\u5bb9\u91cf\u9650\u5236\uff1a" + name);
            if (!seen.emplace(entry).second) throw std::runtime_error(u8"\u586b\u5145\u65cf\u542b\u91cd\u590d\u9879\uff1a" + name);
            slot.width = U(entry.size());
            ++slot.radix;
            for (const auto b : entry) tables.push_back(static_cast<unsigned char>(b));
        };
        const auto chars = [&](std::string_view text) {
            for (size_t at = 0; at < text.size();) {
                const auto begin = at;
                read_scalar(text, at);
                append(text.substr(begin, at - begin));
            }
        };
        if (src != sources.end()) {
            if (src->second.file) {
                std::ifstream file(path.parent_path() / fs::u8path(src->second.value), std::ios::binary);
                if (!file) throw std::runtime_error(u8"\u65e0\u6cd5\u6253\u5f00\u586b\u5145\u8bcd\u5e93\uff1a" + name);
                std::string word;
                bool at_start = true;
                while (config_line(file, word, at_start)) {
                    for (size_t i = 0; i < word.size();) read_scalar(word, i);
                    append(word);
                }
            } else chars(src->second.value);
        } else if (name == "digits") chars("0123456789");
        else if (name == "lower") chars("abcdefghijklmnopqrstuvwxyz");
        else if (name == "upper") chars("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
        else if (name == "alnum") chars("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
        else if (name == "hanzi") {
            for (U c = 0x4e00; c <= 0x9fff; ++c) {
                const char bytes[] = {char(0xe0 | (c >> 12)), char(0x80 | ((c >> 6) & 63)), char(0x80 | (c & 63))};
                append(std::string_view(bytes, 3));
            }
        } else throw std::runtime_error(u8"\u672a\u5b9a\u4e49\u586b\u5145\u65cf\uff1a" + name);
        if (!slot.radix) throw std::runtime_error(u8"\u586b\u5145\u65cf\u6ca1\u6709\u6761\u76ee\uff1a" + name);
        loaded.emplace(name, slot);
        return slot;
    };
    plan.domain = 1;
    for (size_t at = 0; at < format->size();) {
        if ((*format)[at] == '{') {
            const auto end = format->find('}', at + 1);
            const auto name = format->substr(at + 1, end == format->npos ? end : end - at - 1);
            if (end == format->npos || !fill_family(name)) throw std::runtime_error(u8"\u65e0\u6548\u7684 {\u586b\u5145\u65cf} \u5f15\u7528");
            auto slot = get_slot(name);
            slot.offset = U(plan.data.size());
            plan.data.resize(plan.data.size() + slot.width);
            plan.slots.push_back(slot);
            if (plan.domain > std::numeric_limits<Id>::max() / slot.radix)
                throw std::runtime_error(u8"\u6df7\u5408\u5b57\u7b26\u96c6\u5bb9\u91cf\u8d85\u8fc7 u64\uff1b\u8bf7\u51cf\u5c11\u53ef\u53d8\u4f4d\u7f6e");
            plan.domain *= slot.radix;
            at = end + 1;
        } else {
            const auto begin = at;
            if (read_scalar(*format, at) == '}') throw std::runtime_error(u8"\u586b\u5145 format \u542b\u672a\u914d\u5bf9\u7684 } \u5b57\u7b26");
            for (auto i = begin; i < at; ++i) plan.data.push_back(static_cast<unsigned char>((*format)[i]));
        }
        if (plan.data.size() > name_limit) throw std::runtime_error(u8"\u586b\u5145\u957f\u5ea6\u8d85\u8fc7 252 \u5b57\u8282");
    }
    if (plan.slots.empty()) throw std::runtime_error(u8"\u586b\u5145 format \u81f3\u5c11\u9700\u8981\u4e00\u4e2a\u53ef\u53d8\u4f4d\u7f6e");
    plan.length = int(plan.data.size());
    for (auto& slot : plan.slots) slot.table += U(plan.length);
    plan.data.insert(plan.data.end(), tables.begin(), tables.end());
    return plan;
}

static Id fill_decode(const Launch& input, std::string_view text) {
    Id id = 0;
    if (!input.fill_slots_count) {
        const bool legacy = text.size() == 18 && text.substr(15) == u8"\u3d01";
        if (!legacy && text.size() != encoded_width) throw std::runtime_error(u8"Base16384 \u586b\u5145\u957f\u5ea6\u4e0d\u7b26");
        size_t at = 0;
        for (int i = 0; i < 5; ++i) {
            const U c = read_scalar(text, at) - 0x4e00u;
            if (c >= 16384 || (i == 4 && (c & 63))) throw std::runtime_error(u8"Base16384 \u586b\u5145\u503c\u65e0\u6548");
            id = i == 4 ? (id << 8) | (c >> 6) : (id << 14) | c;
        }
        if (legacy) return id; // 旧末尾标志对应旧顺序编号，不应用新置换。
    } else {
        if (text.size() != size_t(input.fill_len)) throw std::runtime_error(u8"\u586b\u5145\u957f\u5ea6\u4e0e\u5e03\u5c40\u4e0d\u7b26");
        U at = 0;
        for (U i = 0; i < input.fill_slots_count; ++i) {
            const auto slot = input.fill_slots[i];
            for (; at < slot.offset; ++at)
                if (U(static_cast<unsigned char>(text[at])) != input.fill_data[at])
                    throw std::runtime_error(u8"\u586b\u5145\u5b57\u9762\u5185\u5bb9\u4e0e\u5e03\u5c40\u4e0d\u7b26");
            U digit = 0;
            for (; digit < slot.radix; ++digit) {
                const auto* word = input.fill_data + slot.table + digit * slot.width;
                if (std::equal(word, word + slot.width, text.begin() + at,
                               [](U a, unsigned char b) { return a == b; })) break;
            }
            if (digit == slot.radix) throw std::runtime_error(u8"\u586b\u5145\u6761\u76ee\u4e0d\u5728\u5b57\u7b26\u96c6\u4e2d");
            id = id * slot.radix + digit;
            at += slot.width;
        }
        for (; at < text.size(); ++at)
            if (U(static_cast<unsigned char>(text[at])) != input.fill_data[at])
                throw std::runtime_error(u8"\u586b\u5145\u5b57\u9762\u5185\u5bb9\u4e0e\u5e03\u5c40\u4e0d\u7b26");
    }
    return shuffle_id<true>(input, id);
}

static Launch prepare(const Options& opts, const Rules& plan, const FillPlan& fill) {
    const auto first = opts.pattern.find("[]");
    if (first == opts.pattern.npos || opts.pattern.find("[]", first + 2) != opts.pattern.npos
        || opts.pattern.size() + fill.length - 2 > name_limit || opts.team.size() > 256
        || opts.pattern.find('\0') != opts.pattern.npos || opts.team.find('\0') != opts.team.npos)
        throw std::runtime_error(u8"PBB \u6a21\u677f\u9700\u8981\u4e00\u4e2a []\uff0c\u751f\u6210\u7ed3\u679c\u6700\u591a 252 \u5b57\u8282\uff0c\u961f\u540d\u6700\u591a 256 \u5b57\u8282\uff0c\u5747\u4e0d\u542b NUL");
    Launch input{};
    input.rules = plan;
    input.fill_len = fill.length;
    input.fill_seed = opts.seed.value_or(0);
    input.fill_domain = fill.domain;
    input.fill_shuffle = opts.shuffle;
    input.fill_slots_count = U(fill.slots.size());
    input.fill_data_count = U(fill.data.size());
    input.fill_slots = fill.slots.data();
    input.fill_data = fill.data.data();
    Id mask = fill.domain ? fill.domain - 1 : std::numeric_limits<Id>::max();
    U bits = 0;
    for (Id v = mask; v; v >>= 1) ++bits;
    mask = bits == 64 ? std::numeric_limits<Id>::max() : (Id(1) << bits) - 1;
    input.fill_shift = std::max(1u, bits / 2);
    input.fill_mask_lo = U(mask);
    input.fill_mask_hi = U(mask >> 32);
    const auto suffix = opts.pattern.substr(first + 2);
    input.prefix_len = int(first);
    input.suffix_len = int(suffix.size());
    input.name_len = input.prefix_len + input.fill_len + input.suffix_len;
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
