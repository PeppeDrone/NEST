// NEST project page: restricted-asset fallback, command selector and synchronised video groups.
// No external dependencies.

(function () {
  "use strict";

  // ---- restricted assets: show a labelled placeholder when the file is not present (public deployment)
  function markMissing(el) {
    var box = el.closest(".restricted");
    if (box) box.classList.add("is-missing");
    var demos = el.closest("#demonstrations");
    if (demos) demos.classList.add("demos-missing");
  }
  function watchMedia(root) {
    root.querySelectorAll(".restricted img").forEach(function (img) {
      if (img.complete && img.naturalWidth === 0) markMissing(img);
      img.addEventListener("error", function () { markMissing(img); });
    });
    root.querySelectorAll(".restricted video").forEach(function (video) {
      video.addEventListener("error", function () { markMissing(video); }, true);
      var src = video.querySelector("source");
      if (src) src.addEventListener("error", function () { markMissing(video); });
    });
  }

  // ---- synchronised groups: all videos with the same data-sync restart together
  function syncGroup(name) {
    var videos = Array.prototype.slice.call(document.querySelectorAll('video[data-sync="' + name + '"]'));
    if (!videos.length) return null;
    var lead = videos[0];
    function align() {
      videos.forEach(function (v) {
        if (v !== lead && Math.abs(v.currentTime - lead.currentTime) > 0.08) v.currentTime = lead.currentTime;
      });
    }
    lead.addEventListener("timeupdate", align);
    lead.addEventListener("ended", function () {
      videos.forEach(function (v) { v.currentTime = 0; v.play().catch(function () {}); });
    });
    videos.forEach(function (v) { if (v !== lead) v.loop = false; });
    lead.loop = false;
    return {
      restart: function () { videos.forEach(function (v) { v.currentTime = 0; v.play().catch(function () {}); }); },
      toggle: function () {
        var paused = lead.paused;
        videos.forEach(function (v) { if (paused) v.play().catch(function () {}); else v.pause(); });
      }
    };
  }

  document.addEventListener("DOMContentLoaded", function () {
    watchMedia(document);

    // ---- figures open at full size in a new tab (useful on small screens)
    document.querySelectorAll(".paper-figure > img").forEach(function (img) {
      img.style.cursor = "zoom-in";
      img.title = "Open full-size figure";
      img.addEventListener("click", function () { window.open(img.currentSrc || img.src, "_blank", "noopener"); });
    });

    var groups = {};
    ["pair", "grid", "teaser"].forEach(function (g) { groups[g] = syncGroup(g); });
    document.querySelectorAll("[data-sync-action]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var g = groups[btn.getAttribute("data-group")];
        if (g) g[btn.getAttribute("data-sync-action")]();
      });
    });

    // ---- command selector for the manual vs automatic pair
    var buttons = document.querySelectorAll(".cmd-buttons .button");
    var prompt = document.getElementById("demo-prompt");
    var vids = {
      manual: document.getElementById("demo-manual"),
      automatic: document.getElementById("demo-automatic")
    };
    function select(btn) {
      buttons.forEach(function (b) { b.classList.remove("is-selected"); });
      btn.classList.add("is-selected");
      var cmd = btn.getAttribute("data-cmd");
      if (prompt) prompt.textContent = "“" + btn.getAttribute("data-prompt") + "”";
      Object.keys(vids).forEach(function (arm) {
        var v = vids[arm];
        if (!v) return;
        var box = v.closest(".restricted");
        if (box) box.classList.remove("is-missing");
        v.src = "./static/restricted/videos/commands/" + arm + "_" + cmd + ".mp4";
        v.load();
        v.play().catch(function () {});
      });
    }
    buttons.forEach(function (btn) { btn.addEventListener("click", function () { select(btn); }); });
    Object.keys(vids).forEach(function (arm) {
      if (vids[arm]) vids[arm].addEventListener("error", function () { markMissing(vids[arm]); });
    });
  });
})();
