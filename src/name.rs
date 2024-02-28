
pub fn median(x: u8, y: u8, z: u8) -> u8 {
    // std::max(std::min(x, y), std::min(std::max(x, y), z))
    x.max(y).max(x.min(y).min(z))
}

/*
function Poly(x) {
    var xp = new Array()
    for (let y = 0; y < 1034; y++) {
        var l = 44
        var i = 0, p = 0, q = 0, r = 0
        var j = y
        for (let k = 0; k < 45; k++) {
            i++;
            if (i > 2) p++;
            q = j;
            j = j - l + p;
            if (j < 0) break;
        }
        if (i == 1) r = x[q]
        if (i > 1) {
            r = x[p] * x[p + q]
        }
        xp[y] = r
    }
    return xp
}
function onStart() {
    var tmp1 = document.getElementById("input").value.trim()
    var names = Array.prototype.slice.call(tmp1.split('\n'));

    var output = document.getElementById("output")
    var dis = document.getElementById("dis")
    output.value = ''

    var tmpsize = parseInt(document.getElementById("tmpsize").value.trim())
    var lim = parseInt(document.getElementById("lim").value.trim())
    if (isNaN(lim)) lim = 0

    var x = new Array(43)
    var name = new Name()
    var s = 0, tmp2 = 0, tmp3 = ''
    var length = names.length
    var Loop = setInterval(function () {
        tmp3 = ''
        for (let ii = 0; ii < tmpsize; ii++) {
            s = tmp2 + ii
            var nametmp = Array.prototype.slice.call(names[s].split('@'));
            if (nametmp.length < 2) nametmp[1] = nametmp[0]
            name.load_team(nametmp[1])
            name.load_name(nametmp[0])
            if (nametmp[1] == "!") name.TV()
            var props = name.calc_props()
            name.calc_skills()
            for (let j = 0; j < 7; j++)props[j] += 36;
            x = new Array(44)

            x[0] = props[7]
            for (let i = 0; i < 7; i++) {
                x[i + 1] = props[i]
            }
            for (let i = 0; i < 35; i++) {
                var cf = 0;
                for (let k = 0; k < 16; k++) {
                    if (name.skill[k] == i) {
                        x[i + 8] = name.freq[k]
                        cf = 1;
                    }
                }
                if (cf == 0) {
                    x[i + 8] = 0
                }
            }
            if (x[32] > 0) {//x[32]>48
                name.load_name(nametmp[0] + '?shadow')
                props = name.calc_props()
                var shadow_sum = props[7] / 3
                for (let j = 0; j < 7; j++)shadow_sum += props[j]

                //更新部分
                shadow_sum -= props[6] * 3
                var shadowi = shadow_sum - 210

                //更新部分
                shadowi = shadowi * x[32] / 100
                x[43] = shadowi.toFixed(3)
            } else {
                x[43] = 0
            }
            if (x[42] > 0) x[42] += 20

            var xp = Poly(x)
            var score = model[0]
            var scoreQD = modelQD[0]
            for (let i = 0; i < 1034; i++) {
                score += xp[i] * model[i + 1]
            }
            for (let i = 0; i < 1034; i++) {
                scoreQD += xp[i] * modelQD[i + 1]
            }

            if (score >= lim && x[32] > 48) {
                tmp3 += names[s] + ' ' + parseInt(score) + ' ' + parseInt(scoreQD) + ' !幻术\n'
            } else if (score >= lim && x[26] > 48) {
                tmp3 += names[s] + ' ' + parseInt(score) + ' ' + parseInt(scoreQD) + ' !铁壁\n'
            } else if (score >= lim && x[29] > 48) {
                tmp3 += names[s] + ' ' + parseInt(score) + ' ' + parseInt(scoreQD) + ' !背刺\n'
            } else if (score >= lim && x[11] > 48) {
                tmp3 += names[s] + ' ' + parseInt(score) + ' ' + parseInt(scoreQD) + ' !地裂\n'
            } else if (score >= lim && x[20] > 48) {
                tmp3 += names[s] + ' ' + parseInt(score) + ' ' + parseInt(scoreQD) + ' !加速\n'
            } else if (score >= lim) { tmp3 += names[s] + ' ' + parseInt(score) + ' ' + parseInt(scoreQD) + '\n' }

            names[s] = null
            s++
            if (ii == tmpsize - 1 || s == length) {
                dis.innerText = (s) + ' / ' + length
                output.value += tmp3
            }
            if (s == length) {
                dis.innerText = "测试完成"
                clearInterval(Loop)
                break
            }
        }
        tmp2 += tmpsize
    }, 0)
}
function LoadVersion() {
    var dis = document.getElementById("dis")
    dis.innerText = "模型版本： " + version
}
 */