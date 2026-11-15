"""Linear Algebra Image Compression System — single-file Python web application.
Run: python svd_image_compression_project.py
Dependencies: pip install numpy pillow
Open: http://127.0.0.1:8001
"""
from __future__ import annotations
import base64, csv, io, json, math, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Timer
import numpy as np
from PIL import Image

HOST, PORT, SIZE = "127.0.0.1", 8001, 256
RANKS = (5, 20, 50, 100)
SHEAR = 0.08

# Convert any uploaded image into a square 256x256 grayscale pixel matrix.
def image_to_matrix(image: Image.Image, size: int = SIZE) -> np.ndarray:
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    image = image.crop(((width-side)//2, (height-side)//2, (width+side)//2, (height+side)//2))
    return np.asarray(image.resize((size, size), Image.Resampling.LANCZOS).convert("L"), dtype=np.float64)

# Make a repeatable synthetic image so the application can be tested without uploading a file.
def generate_pattern(size: int = SIZE) -> np.ndarray:
    rows, cols = np.mgrid[0:size, 0:size].astype(float)
    return np.clip(128 + 55*np.sin(cols/size*4*np.pi)*np.cos(rows/size*4*np.pi) + 22*np.sin((rows+cols)/9), 0, 255)

# Apply the slide's shear transformation T = I + 0.08 times the super-diagonal.
def shear_transform(matrix: np.ndarray) -> np.ndarray:
    transform = np.eye(matrix.shape[0])
    transform[np.arange(matrix.shape[0]-1), np.arange(1, matrix.shape[0])] = SHEAR
    return np.clip(transform @ matrix, 0, 255)

# Encode a numeric image matrix as a PNG data URL for display in the browser.
def matrix_to_data_url(matrix: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.clip(matrix, 0, 255).astype(np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

# Decode an uploaded browser data URL into a PIL image.
def data_url_to_image(data_url: str) -> Image.Image:
    payload = data_url.split(",", 1)[-1]
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")

# Perform Gaussian elimination on a small square matrix and return its row-echelon form.
def gaussian_elimination(matrix: np.ndarray) -> dict:
    a = np.array(matrix, dtype=float, copy=True)
    rows, cols = a.shape
    pivot_row = 0
    swaps = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        best = pivot_row + int(np.argmax(np.abs(a[pivot_row:, col])))
        if abs(a[best, col]) < 1e-10:
            continue
        if best != pivot_row:
            a[[pivot_row, best]] = a[[best, pivot_row]]
            swaps += 1
        a[pivot_row] /= a[pivot_row, col]
        for row in range(pivot_row + 1, rows):
            a[row] -= a[row, col] * a[pivot_row]
        pivot_row += 1
    return {"echelon": a.tolist(), "rank": int(np.linalg.matrix_rank(matrix)), "row_swaps": swaps}

# Compute an LU factorization with partial pivoting so P@A approximately equals L@U.
def lu_decomposition(matrix: np.ndarray) -> dict:
    a = np.array(matrix, dtype=float, copy=True)
    n, m = a.shape
    if n != m:
        raise ValueError("LU decomposition requires a square matrix.")
    u, l, p = a.copy(), np.eye(n), np.eye(n)
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(u[col:, col])))
        if abs(u[pivot, col]) < 1e-10:
            continue
        if pivot != col:
            u[[col, pivot]] = u[[pivot, col]]
            p[[col, pivot]] = p[[pivot, col]]
            if col:
                l[[col, pivot], :col] = l[[pivot, col], :col]
        for row in range(col + 1, n):
            factor = u[row, col] / u[col, col]
            l[row, col] = factor
            u[row, col:] -= factor * u[col, col:]
    return {"L": l.tolist(), "U": u.tolist(), "P": p.tolist(), "reconstruction_error": float(np.linalg.norm(p @ a - l @ u))}

# Use Gauss-Jordan elimination to calculate an inverse, returning null when the matrix is singular.
def gauss_jordan_inverse(matrix: np.ndarray) -> dict:
    a = np.array(matrix, dtype=float, copy=True)
    n, m = a.shape
    if n != m:
        raise ValueError("Inverse requires a square matrix.")
    aug = np.hstack((a, np.eye(n)))
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(aug[col:, col])))
        if abs(aug[pivot, col]) < 1e-10:
            return {"inverse": None, "message": "Matrix is singular; inverse does not exist."}
        aug[[col, pivot]] = aug[[pivot, col]]
        aug[col] /= aug[col, col]
        for row in range(n):
            if row != col:
                aug[row] -= aug[row, col] * aug[col]
    return {"inverse": aug[:, n:].tolist(), "message": "Inverse computed by Gauss-Jordan elimination."}

# Derive singular values from the eigenvalues of A-transpose-A for the custom small-matrix SVD.
def custom_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(matrix, dtype=float)
    eigenvalues, v = np.linalg.eigh(a.T @ a)
    order = np.argsort(eigenvalues)[::-1]
    singular_values = np.sqrt(np.clip(eigenvalues[order], 0, None))
    v = v[:, order]
    u = np.zeros((a.shape[0], len(singular_values)))
    nonzero = singular_values > 1e-10
    u[:, nonzero] = (a @ v[:, nonzero]) / singular_values[nonzero]
    # Keep each left singular vector paired with its corresponding singular value and right vector.
    return u, singular_values, v.T

# Compute low-rank SVD reconstructions and the quality/storage metrics used in the slides.
def analyze_svd(matrix: np.ndarray) -> dict:
    start = time.perf_counter()
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    total_energy = float(np.sum(singular_values**2)) or 1.0
    cumulative_energy = np.cumsum(singular_values**2) / total_energy
    results = []
    n, m = matrix.shape
    original_norm = float(np.linalg.norm(matrix)) or 1.0
    for k in RANKS:
        effective_k = min(k, len(singular_values))
        reconstructed = np.clip((u[:, :effective_k] * singular_values[:effective_k]) @ vt[:effective_k, :], 0, 255)
        mse = float(np.mean((matrix - reconstructed)**2))
        psnr = float("inf") if mse == 0 else 10 * math.log10(255**2 / mse)
        # A rank-k matrix needs k*(rows + columns + one singular value) stored numbers.
        compressed_values = effective_k * (n + m + 1)
        ratio = (n*m) / compressed_values
        results.append({"k": effective_k, "image": matrix_to_data_url(reconstructed), "ratio": ratio,
                        "error": float(np.linalg.norm(matrix-reconstructed) / original_norm),
                        "mse": mse, "psnr": psnr, "energy": float(cumulative_energy[effective_k-1]*100),
                        "storage_percent": float(compressed_values/(n*m)*100)})
    numerical_rank = int(np.sum(singular_values > max(matrix.shape) * np.finfo(float).eps * singular_values[0]))
    matrix_methods = analyze_matrix_methods(matrix, singular_values)  # Feed the slide's matrix-method results into the main dashboard.
    return {"results": results, "singular_values": singular_values.tolist(),
            "rank": numerical_rank, "matrix_analysis": matrix_methods,
            "seconds": time.perf_counter()-start,
            "original": matrix_to_data_url(matrix), "transformed": matrix_to_data_url(shear_transform(matrix)),
            "shape": list(matrix.shape)}

# Run the slide's four matrix-analysis methods on a 4x4 patch and compute rank for the full image matrix.
def analyze_matrix_methods(matrix: np.ndarray, singular_values: np.ndarray | None = None) -> dict:
    patch = np.asarray(matrix[:4, :4], dtype=float) / 255.0  # Normalize a small pixel region for readable matrix values.
    patch = np.round(patch, 5)
    gaussian = gaussian_elimination(patch)  # Demonstrate row operations and row-echelon form on the 4x4 patch.
    lu = lu_decomposition(patch)  # Factor the patch into P, L and U and measure P·A − L·U residual.
    inverse = gauss_jordan_inverse(patch)  # Test whether the chosen image patch has a valid Gauss–Jordan inverse.
    patch_rank = int(np.linalg.matrix_rank(patch))  # Count independent rows/columns in the local image patch.
    det = float(np.linalg.det(patch))  # Calculate the determinant as another invertibility check.
    if singular_values is None:
        singular_values = np.linalg.svd(matrix, compute_uv=False)  # Obtain singular values when this function is called alone.
    tol = max(matrix.shape) * np.finfo(float).eps * (float(singular_values[0]) if len(singular_values) else 0.0)
    full_rank = int(np.sum(singular_values > tol))  # Numerical rank of the complete 256x256 grayscale image matrix.
    energy = np.cumsum(singular_values ** 2) / max(float(np.sum(singular_values ** 2)), 1e-30)
    energy_ranks = {str(pct): int(np.searchsorted(energy, pct / 100.0) + 1) for pct in (90, 95, 99)}
    return {
        "patch": patch.tolist(), "gaussian": gaussian, "lu": lu,
        "full_matrix_shape": list(matrix.shape), "full_rank": full_rank,
        "full_row_rank": full_rank, "full_column_rank": full_rank,
        "full_rank_max": min(matrix.shape), "effective_rank_by_energy": energy_ranks,
        "patch_rank": patch_rank, "patch_rank_max": min(patch.shape),
        "patch_rows_independent": patch_rank == patch.shape[0],
        "patch_columns_independent": patch_rank == patch.shape[1],
        "determinant": det, "inverse": inverse,
        "inverse_exists": inverse["inverse"] is not None,
        "lu_reconstruction_error": lu["reconstruction_error"],
    }

PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Linear Algebra Image Compression</title>
<style>
:root{--bg:#f6f4ee;--ink:#191612;--muted:#777064;--line:#e4dece;--card:#fff;--accent:#b96f3d}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,Arial}header{position:sticky;top:0;background:#f6f4eef2;border-bottom:1px solid var(--line);padding:14px 4%;z-index:2;display:flex;align-items:center;gap:12px}.logo{background:var(--ink);color:white;border-radius:50%;padding:5px 11px;font-size:20px}main{max-width:1300px;margin:auto;padding:26px 4% 60px}h1{font-size:clamp(28px,4vw,44px);line-height:1.1;margin:14px 0}h2{font-size:22px;margin:0 0 12px}p,.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:15px}.card{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:17px;margin:14px 0}.drop{border:2px dashed #c9beaa;text-align:center;padding:22px;border-radius:12px;cursor:pointer}.btn{border:1px solid var(--line);background:white;border-radius:8px;padding:10px 14px;cursor:pointer;font-weight:650}.primary{background:var(--ink);color:white}.btn:disabled{opacity:.5}.preview{width:100%;aspect-ratio:1;object-fit:contain;background:#eee;border-radius:8px}.small{font-size:12px;color:var(--muted)}.tag{font:11px Consolas,monospace;color:var(--muted);text-transform:uppercase;letter-spacing:.12em}.metric{font-size:23px;font-weight:700}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font:13px Consolas,monospace}td,th{text-align:right;padding:10px;border-bottom:1px solid var(--line);white-space:nowrap}td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8f6f0;border-radius:8px;padding:12px;font-size:12px}.bar{height:8px;background:#eee8db;border-radius:20px;overflow:hidden}.bar span{display:block;height:100%;background:var(--accent)}canvas{width:100%;height:250px}.hidden{display:none}.status{margin-left:auto;font-size:12px;color:var(--muted)}.method-card{background:#fbfaf6;border:1px solid var(--line);border-radius:12px;padding:14px;min-width:0}.method-card b{display:block;margin:6px 0}.method-card pre{max-height:250px;overflow:auto}.method-card .metric{font-size:20px}summary{cursor:pointer}footer{color:var(--muted);font-size:12px;margin-top:30px}@media(max-width:600px){header{padding:12px}main{padding:18px 12px}}
</style></head><body><header><div class="logo">Σ</div><div><b>Linear Algebra Lab</b><div class="small">Image compression · SVD · matrix methods</div></div><div id="status" class="status">Ready</div></header><main>
<div class="tag">Mathematics mini project</div><h1>Image Compression System</h1><p>Convert a grayscale image into a matrix, investigate its structure, and compare low-rank SVD reconstructions.</p>
<div class="grid"><div class="card"><h2>1. Choose an image</h2><label class="drop" id="drop"><b>Click or drop an image here</b><div class="small">JPG, PNG or BMP · processed to 256 × 256</div><input id="file" type="file" accept="image/png,image/jpeg,image/bmp" hidden></label><div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap"><button class="btn" id="pattern">Generate test pattern</button><button class="btn primary" id="run">Run all analysis</button></div></div><div class="card"><h2>2. Matrix representation</h2><div class="metric">A ∈ ℝ²⁵⁶ˣ²⁵⁶</div><p class="small">Each entry Aᵢⱼ is a grayscale pixel intensity between 0 and 255.</p><div class="tag">Pipeline</div><p>Image → grayscale matrix → matrix analysis → shear transform → SVD → reconstruction → metrics</p></div></div>
<section class="card"><h2>Step 1–2 · Image and shear transformation</h2><div id="images" class="grid"><p class="muted">Load an image or generate a pattern, then run the analysis.</p></div></section>
<section class="card"><h2>Step 3 — Matrix operations</h2><p class="small">The four methods from the presentation run automatically when you click “Run all analysis”. Gaussian elimination and LU use a 4 × 4 pixel patch; rank is also measured on the full 256 × 256 image matrix.</p><div class="grid matrix-cards"><div class="method-card" style="border-left:5px solid #20cbd0"><div class="tag">01 · Gaussian elimination</div><b>Simplify the 4 × 4 patch</b><p class="small">Use row operations to obtain row-echelon form and expose pivot positions.</p><div id="gaussianOutput"><p class="muted">Run analysis to see the result.</p></div></div><div class="method-card" style="border-left:5px solid #53d6b5"><div class="tag">02 · LU decomposition</div><b>Factor the pixel patch</b><p class="small">Find P, L and U such that P·A = L·U.</p><div id="luOutput"><p class="muted">Run analysis to see the result.</p></div></div><div class="method-card" style="border-left:5px solid #f0b33b"><div class="tag">03 · Rank</div><b>Inspect the full image matrix</b><p class="small">Numerically count independent rows and columns of the 256 × 256 matrix.</p><div id="rankOutput"><p class="muted">Run analysis to see the result.</p></div></div><div class="method-card" style="border-left:5px solid #578cff"><div class="tag">04 · Inverse & independence</div><b>Check the 4 × 4 patch</b><p class="small">Gauss–Jordan tests invertibility; rank indicates whether its rows and columns are independent.</p><div id="inverseOutput"><p class="muted">Run analysis to see the result.</p></div></div></div><details style="margin-top:14px"><summary class="btn">Show full matrix calculations</summary><div id="matrixOutput"><p class="muted">Detailed matrices will appear after analysis.</p></div></details></section>
<section class="card"><h2>Step 4–5 · SVD compression and reconstruction</h2><p class="small">A = UΣVᵀ. Keep only the first k singular values and corresponding vectors to form Aₖ = UₖΣₖVₖᵀ.</p><div id="recon" class="grid"><p class="muted">Results appear after analysis.</p></div></section>
<section class="card"><h2>Compression–quality comparison</h2><div class="scroll"><table><thead><tr><th>Method</th><th>k</th><th>Ratio</th><th>Relative error</th><th>MSE</th><th>PSNR (dB)</th><th>Energy</th><th>Storage</th></tr></thead><tbody id="tbody"><tr><td colspan="8">No results yet</td></tr></tbody></table></div><div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:14px"><button class="btn" id="csv" disabled>Export CSV</button><button class="btn" id="png" disabled>Export selected reconstruction</button><select id="exportK" class="btn"><option value="5">Export k = 5</option><option value="20" selected>Export k = 20</option><option value="50">Export k = 50</option><option value="100">Export k = 100</option></select></div></section>
<section class="card"><h2>Singular-value decay</h2><p class="small">The curve shows singular values in descending order; larger values represent stronger matrix directions.</p><canvas id="chart" width="900" height="280"></canvas><div id="summary" class="grid"></div></section>
<section class="card"><h2>Custom SVD verification</h2><p class="small">A from-scratch eigenvalue-based SVD is tested on a small 4 × 4 patch and compared with NumPy's SVD reconstruction.</p><button class="btn" id="customBtn">Run custom SVD check</button><div id="customOutput"></div></section>
<footer>Educational demonstration. Compression ratio compares the number of pixel values in the original matrix with the number of values in the low-rank factor representation; actual encoded file size may differ.</footer></main>
<script>
const $=id=>document.getElementById(id);let imageData=null,report=null,matrixReport=null;
function status(t){$('status').textContent=t}async function post(path,body={}){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw Error(j.error||'Request failed');return j}
function imageCard(title,src,sub=''){return `<div><div class="tag">${title}</div><img class="preview" src="${src}" alt="${title}"><div class="small">${sub}</div></div>`}
$('drop').onclick=()=> $('file').click();$('file').onchange=async e=>{const f=e.target.files[0];if(!f)return;imageData=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsDataURL(f)});status('Image loaded · ready to analyze');};$('drop').ondragover=e=>e.preventDefault();$('drop').ondrop=async e=>{e.preventDefault();const f=e.dataTransfer.files[0];if(f){imageData=await new Promise(resolve=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.readAsDataURL(f)});status('Image loaded · ready to analyze')}};
$('pattern').onclick=async()=>{imageData=null;await runAll()};$('run').onclick=runAll;
async function runAll(){try{status('Computing SVD and matrix methods…');$('run').disabled=true;report=await post('/api/analyze',{image:imageData});renderMatrixMethods(report.matrix_analysis);$('images').innerHTML=imageCard('Original grayscale',report.original,'256 × 256 pixel matrix')+imageCard('Shear transformation',report.transformed,'T = I + 0.08 super-diagonal');$('recon').innerHTML=report.results.map(r=>`<div>${imageCard('SVD · k = '+r.k,r.image,`${r.ratio.toFixed(2)}× ratio · ${r.psnr.toFixed(2)} dB PSNR`)}<div class="bar"><span style="width:${Math.min(100,r.energy)}%"></span></div><div class="small">Retained energy: ${r.energy.toFixed(3)}%</div></div>`).join('');$('tbody').innerHTML=report.results.map(r=>`<tr><td>SVD</td><td>${r.k}</td><td>${r.ratio.toFixed(2)}×</td><td>${r.error.toFixed(5)}</td><td>${r.mse.toFixed(3)}</td><td>${r.psnr.toFixed(2)}</td><td>${r.energy.toFixed(3)}%</td><td>${r.storage_percent.toFixed(2)}%</td></tr>`).join('');$('summary').innerHTML=`<div><div class="tag">Numerical rank</div><div class="metric">${report.rank}</div></div><div><div class="tag">Matrix size</div><div class="metric">256 × 256</div></div><div><div class="tag">SVD runtime</div><div class="metric">${report.seconds.toFixed(3)} s</div></div>`;drawChart(report.singular_values);$('csv').disabled=false;$('png').disabled=false;status('Analysis complete');}catch(e){status('Error: '+e.message);alert(e.message)}finally{$('run').disabled=false}}
function renderMatrixMethods(m){if(!m)return;$('gaussianOutput').innerHTML=`<div class="small">Pivot rows: ${m.gaussian.rank} · Row swaps: ${m.gaussian.row_swaps}</div><pre>${JSON.stringify(m.gaussian.echelon,null,2)}</pre>`;$('luOutput').innerHTML=`<div class="small">Factorization residual ‖P·A − L·U‖: ${m.lu_reconstruction_error.toExponential(2)}</div><pre>L = ${JSON.stringify(m.lu.L)}\nU = ${JSON.stringify(m.lu.U)}\nP = ${JSON.stringify(m.lu.P)}</pre>`;$('rankOutput').innerHTML=`<div class="metric">${m.full_rank} / ${m.full_rank_max}</div><div class="small">Row rank = ${m.full_row_rank}; column rank = ${m.full_column_rank}</div><div class="small">Ranks for 90% / 95% / 99% singular-value energy: ${m.effective_rank_by_energy['90']} / ${m.effective_rank_by_energy['95']} / ${m.effective_rank_by_energy['99']}</div>`;$('inverseOutput').innerHTML=`<div class="small">Patch rank: ${m.patch_rank} / ${m.patch_rank_max}</div><div class="small">Rows independent: ${m.patch_rows_independent?'yes':'no'} · Columns independent: ${m.patch_columns_independent?'yes':'no'}</div><div class="small">Determinant: ${m.determinant.toExponential(3)}</div><div class="small">Inverse exists: ${m.inverse_exists?'yes':'no'}</div><pre>${m.inverse.inverse?JSON.stringify(m.inverse.inverse,null,2):m.inverse.message}</pre>`;$('matrixOutput').innerHTML=`<h3>Normalized 4 × 4 pixel patch</h3><pre>${JSON.stringify(m.patch,null,2)}</pre><div class="grid"><div><b>Gaussian elimination — row-echelon form</b><pre>${JSON.stringify(m.gaussian,null,2)}</pre></div><div><b>LU decomposition (P·A = L·U)</b><pre>${JSON.stringify(m.lu,null,2)}</pre></div><div><b>Full matrix rank and patch independence</b><pre>Full matrix shape: ${m.full_matrix_shape.join(' × ')}\nFull numerical rank: ${m.full_rank}\nFull row rank: ${m.full_row_rank}\nFull column rank: ${m.full_column_rank}\n4 × 4 patch rank: ${m.patch_rank}\nPatch determinant: ${m.determinant}</pre></div><div><b>Gauss–Jordan inverse</b><pre>${JSON.stringify(m.inverse,null,2)}</pre></div></div>`;}
function drawChart(s){const c=$('chart'),ctx=c.getContext('2d'),w=c.width,h=c.height,p={l:48,r:16,t:15,b:35};ctx.clearRect(0,0,w,h);ctx.strokeStyle='#e4dece';ctx.fillStyle='#777064';ctx.font='12px monospace';for(let i=0;i<=4;i++){let y=p.t+(h-p.t-p.b)*i/4;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke();ctx.fillText((1-i/4).toFixed(2),5,y+4)}ctx.strokeStyle='#b96f3d';ctx.lineWidth=2;ctx.beginPath();s.forEach((v,i)=>{let x=p.l+i/(s.length-1)*(w-p.l-p.r),y=p.t+(1-v/s[0])*(h-p.t-p.b);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.fillStyle='#191612';ctx.fillText('Singular value index',w/2-55,h-8)}

$('customBtn').onclick=async()=>{try{const j=await post('/api/custom',{image:imageData});$('customOutput').innerHTML=`<pre>${JSON.stringify(j,null,2)}</pre>`}catch(e){alert(e.message)}};
$('csv').onclick=()=>{if(!report)return;const rows=[['Method','k','Compression ratio','Relative error','MSE','PSNR dB','Energy percent','Storage percent'],...report.results.map(r=>['SVD',r.k,r.ratio,r.error,r.mse,r.psnr,r.energy,r.storage_percent])];const text=rows.map(row=>row.join(',')).join('\n');download(new Blob([text],{type:'text/csv'}),'svd_comparison.csv')};$('png').onclick=()=>{if(!report)return;const r=report.results.find(x=>x.k===+$('exportK').value);const a=document.createElement('a');a.href=r.image;a.download=`svd_reconstruction_k${r.k}.png`;a.click()};function download(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();URL.revokeObjectURL(a.href)}
</script></body></html>'''

# Route browser requests to the embedded user interface and mathematical API functions.
class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    # Serve the self-contained HTML interface and a lightweight health check.
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        elif self.path == "/api/health":
            self.send_json({"ok": True})
        else:
            self.send_error(404)

    # Process uploaded images and return SVD, matrix-method or custom-SVD results as JSON.
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            matrix = image_to_matrix(data_url_to_image(body["image"])) if body.get("image") else generate_pattern()
            if self.path == "/api/analyze":
                result = analyze_svd(matrix)
            elif self.path == "/api/matrix":
                result = analyze_matrix_methods(matrix)
            elif self.path == "/api/custom":
                patch = np.round(matrix[:4, :4] / 255.0, 3)
                uc, sc, vc = custom_svd(patch)
                un, sn, vn = np.linalg.svd(patch, full_matrices=False)
                custom_reconstruction = (uc[:, :len(sc)] * sc) @ vc
                numpy_reconstruction = (un * sn) @ vn
                result = {"custom_singular_values": sc.tolist(), "numpy_singular_values": sn.tolist(),
                          "custom_reconstruction_error": float(np.linalg.norm(patch-custom_reconstruction)),
                          "numpy_reconstruction_error": float(np.linalg.norm(patch-numpy_reconstruction)),
                          "note": "The custom eigenvalue-based method is for a small educational test matrix; NumPy SVD is used for the full image."}
            else:
                self.send_json({"error": "Unknown API endpoint."}, 404); return
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)

    # Serialize a response object as JSON for the browser-side interface.
    def send_json(self, value, status=200):
        data = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

# Start the local server and open its interface in the default browser.
def main():
    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    url = f"http://{HOST}:{PORT}"
    print(f"Linear Algebra Image Compression System running at {url}")
    print("Press Ctrl+C to stop the server.")
    Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
