// Copy button for the install command on the home page.
(function () {
  var btn = document.getElementById('amm-copy');
  if (!btn) return;
  btn.addEventListener('click', function () {
    var cmd = document.getElementById('amm-install-cmd');
    var text = cmd ? cmd.textContent : '';
    function done(msg) {
      btn.textContent = msg;
      setTimeout(function () { btn.textContent = 'Copy'; }, 1500);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { done('Copied'); }, function () { done('Select and copy'); });
    } else {
      done('Select and copy');
    }
  });
})();
