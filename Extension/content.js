function isToxic(predictions) {
  return predictions.toxic > 0.5;
}

function scanComments() {
  // Select unprocessed comments
  const allUnprocessed = document.querySelectorAll("#content-text:not([data-checked])");

  // Convert NodeList to array and pick only first 10
  const commentsToProcess = Array.from(allUnprocessed).slice(0, 10);

  commentsToProcess.forEach(commentNode => {
    const text = commentNode.innerText.trim();

    if (text.length === 0) {
      commentNode.setAttribute("data-checked", "true");
      return;
    }

    chrome.runtime.sendMessage({ type: "CHECK_TOXICITY", comment: text }, (response) => {
      if (!response) {
        console.error("No response from background");
        return;
      }

      if (response.success) {
        const toxic = isToxic(response.data.predictions);
        commentNode.style.backgroundColor = toxic ? "rgba(255, 0, 0, 0.3)" : "rgba(0, 255, 0, 0.2)";
        commentNode.setAttribute("data-checked", "true");
      } else {
        console.error("API error:", response.error);
      }
    });
  });
}

const observer = new MutationObserver(scanComments);
observer.observe(document.body, { childList: true, subtree: true });

window.addEventListener("load", scanComments);
