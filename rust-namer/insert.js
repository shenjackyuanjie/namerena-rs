let has_done_target = document.getElementById('done_target');
if (has_done_target) {
    has_done_target.display = 'none';
    has_done_target.done = 'false';
} else {
    let target = document.createElement("div", style = "display: none;");
    target.id = "done_target";
    target.style.display = "none";
    document.body.appendChild(target);
}
// 监听消息
window.addEventListener('message', function (event) {
    console.log('Received message:', event.data, event);
    if (event.data.hasOwnProperty("all")) {
        // 为 done_target 添加 done=true 属性
        let done_target = document.getElementById('done_target');
        done_target.setAttribute('done', 'true');
        done_target.win_data = event.data;
        console.log('done_target:', done_target);
    }
});