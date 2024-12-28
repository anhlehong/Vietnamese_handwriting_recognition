// Hàm lưu nội dung Markdown vào tệp
function saveToFile() {
  // Lấy nội dung Markdown từ div ẩn
  const markdownContent = document.getElementById("markdownContent").innerText;

  // Tạo một Blob từ nội dung Markdown
  const blob = new Blob([markdownContent], { type: "text/markdown" });

  // Tạo một thẻ <a> để tải xuống
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "result.md"; // Tên tệp mặc định
  link.click();

  // Giải phóng URL sau khi sử dụng
  URL.revokeObjectURL(link.href);
}

document
  .getElementById("saveToObsidian")
  .addEventListener("click", function () {
    // Lấy nội dung Markdown từ div
    const markdownContent = document
      .getElementById("markdownContent")
      .innerText.trim();

    // Kiểm tra nếu không có nội dung
    if (!markdownContent) {
      alert("Không có nội dung để lưu vào Obsidian.");
      return;
    }

    // Hiển thị pop-up để nhập tên tệp
    let fileName = prompt("Nhập tên ghi chú:");

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    fileName = `${fileName}_${timestamp}`;

    // Kiểm tra nếu người dùng không nhập
    if (!fileName) {
      alert("Tên tệp không được để trống.");
      return;
    }

    // Tạo URI Obsidian
    const obsidianURI = `obsidian://new?name=${encodeURIComponent(
      fileName
    )}&content=${encodeURIComponent(markdownContent)}`;

    // Điều hướng tới URI (mở Obsidian)
    window.location.href = obsidianURI;
  });
