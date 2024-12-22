// Hàm lưu nội dung Markdown vào tệp
function saveToFile() {
    // Lấy nội dung Markdown từ div ẩn
    const markdownContent = document.getElementById('markdownContent').innerText;

    // Tạo một Blob từ nội dung Markdown
    const blob = new Blob([markdownContent], { type: 'text/markdown' });

    // Tạo một thẻ <a> để tải xuống
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'result.md'; // Tên tệp mặc định
    link.click();

    // Giải phóng URL sau khi sử dụng
    URL.revokeObjectURL(link.href);
}
