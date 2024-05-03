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

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
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
        var s = 0;
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

const Namer = struct {
    // code from rust
    // pub val: [u8; 256],
    // pub name_base: [u8; 128],
    // pub name_prop: [u32; 8],
    // pub skl_id: [u8; 40],
    // pub skl_freq: [u8; 40],
    val: [256]u8,
    name_base: [128]u8,
    name_prop: [8]u32,
    skl_id: [40]u8,
    skl_freq: [40]u8,

    pub fn new(team: Teamer, name: []const u8) Namer {
        var val = team.val;
        var name_base = [0]u8;
        var name_prop = [0]u32;
        var skl_id = [0]u8;
        var skl_freq = [0]u8;

        const name_len = name.len + 1;
    }
};
