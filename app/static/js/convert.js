document.addEventListener("DOMContentLoaded", () => {
    const markdownContent = document.getElementById('markdownContent').innerText;

    // Sử dụng thư viện `marked` để chuyển Markdown sang HTML
    const htmlOutput = marked.parse(markdownContent);

    // Hiển thị kết quả HTML trong div đầu ra
    document.getElementById('htmlOutput').innerHTML = htmlOutput;
});
