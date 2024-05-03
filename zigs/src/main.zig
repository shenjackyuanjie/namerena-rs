const std = @import("std");

pub fn main() !void {
    // Prints to stderr (it's a shortcut based on `std.io.getStdErr()`)
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});

    // stdout is for the actual output of your application, for example if you
    // are implementing gzip, then only the compressed bytes should be sent to
    // stdout, not any debugging messages.
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    try bw.flush(); // don't forget to flush!
}

const val_init = [256]u8{
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,
    16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
    32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
    48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
    64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
    80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
    96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
};

const Teamer = struct {
    val: [256]u8,

    // 初始化函数: 接受一个字符串参数
    pub fn new(team_name: []const u8) Teamer {
        var val = val_init;
        const t_len = team_name.len + 1;
        var s: u8 = 0;
        for (0..256) |i| {
            if (i % t_len != 0) {
                s +%= team_name[i % t_len - 1];
            }
            s +%= val[i];
            // swap i and s
            const tmp = val[i];
            val[i] = val[s];
            val[s] = tmp;
        }
        return Teamer{ .val = val };
    }
};

fn median(a: u8, b: u8, c: u8) u8 {
    if (a > b) {
        if (b > c) {
            return b;
        } else if (a > c) {
            return c;
        } else {
            return a;
        }
    } else if (a > c) {
        return a;
    } else if (b > c) {
        return c;
    } else {
        return b;
    }
}

const Namer = struct {
    val: [256]u8,
    name_base: [128]u8,
    name_prop: [8]u32,
    skl_id: [40]u8,
    skl_freq: [40]u8,

    pub fn new(team: Teamer, name: []const u8) Namer {
        var val = team.val;
        var name_base = [_]u8{0} ** 128;
        var name_prop = [_]u32{0} ** 8;
        const skl_id = [_]u8{0} ** 40;
        const skl_freq = [_]u8{0} ** 40;

        const name_len = name.len;
        inline for (0..2) |_| {
            var k: u32 = 0;
            var s: u8 = 0;
            inline for (0..256) |i| {
                if (k != 0) {
                    s +%= name[k - 1];
                }
                s +%= val[i];
                // swap i and s
                const tmp = val[i];
                val[i] = val[s];
                val[s] = tmp;
                if (k == name_len) {
                    k = 0;
                } else {
                    k += 1;
                }
            }
        }

        var s: u32 = 0;
        var q_len: i32 = -1;
        inline for (0..96) |i| {
            const m = ((val[i] *% 181) +% 160);
            if (m >= 89 and m < 217) {
                name_base[s] = m & 63;
                s += 1;
                if (q_len == 30) {
                    break;
                } else {
                    q_len += 1;
                }
            }
        }
        if (q_len < 31) {
            inline for (96..256) |i| {
                const m = ((val[i] *% 181) +% 160);
                if (m >= 89 and m < 217) {
                    name_base[s] = m & 63;
                    q_len += 1;
                    s += 1;
                    if (q_len > 30) {
                        break;
                    }
                }
            }
        }

        var prop_name = [_]u8{0} ** 32;
        inline for (0..32) |i| {
            prop_name[i] = name_base[i];
        }
        // sort 0~10
        std.sort.heap(u8, prop_name[0..10], {}, std.sort.asc(u8));
        name_prop[0] = 154 + @as(u32, prop_name[3]) + @as(u32, prop_name[4]) + @as(u32, prop_name[5]) + @as(u32, prop_name[6]);

        name_prop[1] = median(prop_name[10], prop_name[11], prop_name[12]) + 36;
        name_prop[2] = median(prop_name[13], prop_name[14], prop_name[15]) + 36;
        name_prop[3] = median(prop_name[16], prop_name[17], prop_name[18]) + 36;
        name_prop[4] = median(prop_name[19], prop_name[20], prop_name[21]) + 36;
        name_prop[5] = median(prop_name[22], prop_name[23], prop_name[24]) + 36;
        name_prop[6] = median(prop_name[25], prop_name[26], prop_name[27]) + 36;
        name_prop[7] = median(prop_name[28], prop_name[29], prop_name[30]) + 36;

        return Namer{
            .val = val,
            .name_base = name_base,
            .name_prop = name_prop,
            .skl_id = skl_id,
            .skl_freq = skl_freq,
        };
    }
};

test "val_test" {
    // 从 rust 移植过来的测试
    const team = Teamer.new("x");
    const namer = Namer.new(team, "x");

    const test_val_vec = [256]u8{
        225, 96,  49,  232, 20,  47,  115, 245, 234, 23,  111, 178, 231, 100, 118, 197, 42,  98,  137, 196, 209, 86,  114, 184, 167,
        129, 164, 239, 205, 211, 82,  173, 189, 153, 198, 67,  4,   3,   90,  52,  128, 134, 176, 145, 85,  9,   250, 30,  63,  247,
        240, 17,  215, 200, 78,  188, 132, 117, 10,  45,  162, 79,  123, 73,  109, 91,  57,  210, 22,  175, 107, 203, 103, 32,  83,
        70,  242, 75,  220, 140, 148, 15,  138, 44,  228, 43,  105, 199, 99,  116, 97,  69,  80,  172, 230, 25,  224, 33,  31,  135,
        235, 74,  193, 238, 233, 88,  216, 204, 24,  163, 141, 6,   201, 26,  38,  21,  186, 237, 101, 206, 212, 76,  144, 219, 149,
        169, 202, 110, 41,  166, 139, 194, 168, 34,  142, 147, 187, 108, 223, 94,  5,   243, 226, 60,  40,  102, 51,  87,  61,  236,
        46,  159, 64,  227, 113, 190, 81,  127, 65,  8,   183, 253, 150, 249, 229, 37,  156, 182, 180, 246, 124, 244, 174, 122, 89,
        120, 160, 35,  143, 11,  14,  151, 133, 27,  177, 251, 221, 207, 58,  29,  131, 119, 171, 157, 93,  185, 48,  112, 192, 191,
        66,  106, 39,  59,  92,  19,  1,   155, 254, 84,  222, 165, 54,  121, 13,  50,  36,  130, 95,  161, 213, 170, 28,  241, 71,
        53,  68,  218, 0,   252, 16,  136, 179, 158, 248, 2,   154, 12,  125, 126, 255, 18,  146, 104, 77,  152, 208, 214, 72,  55,
        195, 62,  7,   217, 56,  181,
    };

    for (0..256) |i| {
        try std.testing.expectEqual(test_val_vec[i], namer.val[i]);
    }
}

test "name_test" {}
