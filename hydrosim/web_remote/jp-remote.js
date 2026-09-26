// JalPravah on Vercel: workstation status and sign-in bar under the app bar.
(function () {
  const css = `.jp-remote{background:var(--surface);border-bottom:1px solid var(--rule);font-size:13px}
.jp-remote .wrap{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;padding-top:8px;padding-bottom:8px}
.jp-remote .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.jp-remote input{font:13px var(--font-body);padding:5px 8px;border:1px solid var(--rule);border-radius:4px;width:180px}
.jp-remote button{font:600 12.5px var(--font-head);padding:5px 12px;border-radius:4px;border:1px solid var(--d3d);background:var(--d3d);color:#fff;cursor:pointer}
.jp-remote button.quiet{background:transparent;color:var(--muted);border-color:var(--rule)}
.jp-remote .err{color:var(--bad)}
body.jp-offline [data-run],body.jp-offline [data-del],body.jp-offline .confirm,body.jp-offline section.panel:has(#jobs){display:none!important}`;
  document.head.insertAdjacentHTML("beforeend", `<style>${css}</style>`);
  const bar = document.createElement("div");
  bar.className = "jp-remote";
  bar.innerHTML = `<div class="wrap"><span>Checking the JalPravah workstation…</span></div>`;
  const header = document.querySelector("header.appbar");
  header ? header.after(bar) : document.body.prepend(bar);
  const box = bar.firstElementChild;

  async function render() {
    let s;
    try { s = await (await fetch("/api/remote", { cache: "no-store" })).json(); } catch (e) { s = { online: false }; }
    document.body.classList.toggle("jp-offline", !s.online);
    if (!s.online) {
      box.innerHTML = `<span><span class="dot" style="background:var(--muted)"></span><b>Workstation offline</b> · showing the saved results. New scenarios, simulations and satellite mapping work while JalPravah Online is running on the workstation.</span>`;
    } else if (s.signedIn) {
      box.innerHTML = `<span><span class="dot" style="background:var(--ok)"></span><b>Workstation online</b> · signed in: you can build scenarios, run simulations and map floods.</span><button class="quiet" type="button" id="jp-out">Sign out</button>`;
      box.querySelector("#jp-out").onclick = async () => { await fetch("/api/remote/logout"); location.reload(); };
    } else {
      box.innerHTML = `<span><span class="dot" style="background:var(--ok)"></span><b>Workstation online</b> · sign in to build scenarios, run simulations and map floods.</span>
        <form id="jp-login" style="display:flex;gap:8px;align-items:center"><input type="password" id="jp-pw" placeholder="Password" autocomplete="current-password" aria-label="JalPravah password"><button type="submit">Sign in</button><span class="err" id="jp-err"></span></form>`;
      box.querySelector("#jp-login").onsubmit = async (e) => {
        e.preventDefault();
        const r = await fetch("/api/remote/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ password: box.querySelector("#jp-pw").value }) });
        if (r.ok) location.reload(); else box.querySelector("#jp-err").textContent = (await r.json()).error;
      };
    }
  }
  render();
})();
