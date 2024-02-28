use md5::{Digest, Md5};


pub fn median(x: u8, y: u8, z: u8) -> u8 {
    // std::max(std::min(x, y), std::min(std::max(x, y), z))
    x.max(y).max(x.min(y).min(z))
}

/// 核心的一个名字
pub struct Namer {
	pub freq: [u8; 16],
	pub skill: [u8; 40],
	pub val_base: [u8; 256],
	pub name_base: [u8; 128],
	pub val: [u8; 256],
}

impl Namer {
	
}


/*
#ifndef _NAME_H
#define _NAME_H

#include <algorithm>
#include <cstring>

namespace context {
const int N = 256, M = 128, K = 64, skill_cnt = 40, max_len = 25;

typedef unsigned long long u64_t;
typedef unsigned char u8_t;

u8_t freq[16], skill[skill_cnt], val_base[N], name_base[M], val[N];
u8_t p, q, s, name[max_len + 10];
int q_len;
inline u8_t rnd() {
	q += val[++p];
	std::swap(val[p], val[q]);
	u8_t u = val[(val[p] + val[q]) & 255];
	q += val[++p];
	std::swap(val[p], val[q]);
	return (u << 8 | val[(val[p] + val[q]) & 255]) % skill_cnt;
}

void load_team(const char *team) {
	int t_len = strlen(team) + 1;
	u8_t s;
	for (int i = 0; i < N; i++) val_base[i] = i;
	for (int i = s = 0; i < N; ++i) {
		if (i % t_len) s += team[i % t_len - 1];
		s += val_base[i];
		std::swap(val_base[i], val_base[s]);
	}
}

int bw_threshold;

#define median(x, y, z) std::max(std::min(x, y), std::min(std::max(x, y), z))
#define LIM 96
#define WK(x) val[i + x] = val[i + x] * 181 + 160;
#define a name_base

template <int len>
bool load_name(int *st) {
	q_len = -1;
	memcpy(val, val_base, sizeof val);
	for (int _ = 0; _ < 2; _++)
		for (int i = s = 0, j = 0; i < N; i++, j++) {
			s += name[j];
			s += val[i];
			std::swap(val[i], val[s]);
			if (j == len) j = -1;
		}
	for (int i = 0; i < LIM; i += 8) {
		WK(0) WK(1) WK(2) WK(3) WK(4) WK(5) WK(6) WK(7)
	}
	for (int i = 0; i < LIM && q_len < 30; i++)
		if (val[i] >= 89 && val[i] < 217) a[++q_len] = val[i] & 63;

	if (q_len < 30) {
		for (int i = LIM; i < N; i += 8) {
			WK(0) WK(1) WK(2) WK(3) WK(4) WK(5) WK(6) WK(7)
		}
		for (int i = LIM; i < N && q_len < 30; i++)
			if (val[i] >= 89 && val[i] < 217) a[++q_len] = val[i] & 63;
	}

	int V = 252;
	V += median(a[10], a[11], a[12]);
	V += median(a[13], a[14], a[15]);
	V += median(a[16], a[17], a[18]);
	V += median(a[19], a[20], a[21]);
	V += median(a[22], a[23], a[24]);
	V += median(a[25], a[26], a[27]);
	V += median(a[28], a[29], a[30]);
	if (V < bw_threshold - 132) return false;
	std::sort(a, a + 10);
	st[0] = 154 + a[3] + a[4] + a[5] + a[6];
	V += (unsigned)st[0] / 3;
	if (V < bw_threshold) return false;
	st[1] = median(a[10], a[11], a[12]) + 36;
	st[2] = median(a[13], a[14], a[15]) + 36;
	st[3] = median(a[16], a[17], a[18]) + 36;
	st[4] = median(a[19], a[20], a[21]) + 36;
	st[5] = median(a[22], a[23], a[24]) + 36;
	st[6] = median(a[25], a[26], a[27]) + 36;
	st[7] = median(a[28], a[29], a[30]) + 36;
	return true;
}
#undef a

template <int len>
void calc_skills() {
	q_len = -1;
	memcpy(val, val_base, sizeof val);
	for (int _ = 0; _ < 2; _++)
		for (int i = s = 0, j = 0; i < N; i++, j++) {
			s += name[j];
			s += val[i];
			std::swap(val[i], val[s]);
			if (j == len) j = -1;
		}
	for (int i = 0; i < N; i++)
		if (val[i] * 181 + 199 & 128) name_base[++q_len] = val[i] * 53 & 63 ^ 32;

	u8_t *a = name_base + K;
	for (int i = 0; i < skill_cnt; i++) skill[i] = i;
	p = q = s = 0;
	for (int _ = 0; _ < 2; _++)
		for (int i = 0; i < skill_cnt; i++) {
			s = (s + rnd() + skill[i]) % skill_cnt;
			std::swap(skill[i], skill[s]);
		}
	int last = -1;
	for (int i = 0, j = 0; j < 16; i += 4, j++) {
		u8_t p = std::min(std::min(a[i], a[i + 1]), std::min(a[i + 2], a[i + 3]));
		if (p > 10 && skill[j] < 35) {
			freq[j] = p - 10;
			if (skill[j] < 25) last = j;
		} else
			freq[j] = 0;
	}
	if (last != -1) freq[last] <<= 1;
	if (freq[14] && last != 14)
		freq[14] += std::min(std::min(name_base[60], name_base[61]), freq[14]);
	if (freq[15] && last != 15)
		freq[15] += std::min(std::min(name_base[62], name_base[63]), freq[15]);
}

bool (*fp_load_name[max_len + 1])(int *) = {};
void (*fp_calc_skills[max_len + 1])() = {};

template <int i>
void init_table() {
	fp_load_name[i] = load_name<i>;
	fp_calc_skills[i] = calc_skills<i>;
	init_table<i - 1>();
}
template <>
void init_table<-1>() {}

const int __tmp = (init_table<max_len>(), 1);

};	// namespace cal

#endif
 */