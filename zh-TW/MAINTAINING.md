# 維護 SCGSim {#maintaining-scgsim}

本參考頁說明文件建置與套件維護方式。[維護者指南](docs/ownership.qmd)連結至實作與專案參考資料。研究者無須上層工作區，可從[教學](docs/examples.qmd)開始。

## 來源與產生輸出 {#sources-and-generated-output}

`src/scgsim/` 負責可重用產品程式碼。`docs/` 與 README 負責撰寫說明；`_quarto.yml` 負責網站導覽與執行設定。`scripts/` 負責建置及版本工具；`tests/` 負責維護中的套件測試。產生的 HTML、搜尋索引與封裝知識都是衍生輸出。公開元件與可執行 Notebook 來源由 OrPen SC PDK 維護。

## 原生呈現 {#native-presentation}

網站使用完整的上游 Askr 擴充功能，不含本地 CSS 或 JS。目前編輯固定版本為 Askr v0.13.3，修訂版本 `243d3dbdd234004bbe0538b9518fb76027903915`，支援 Quarto >=1.9.38 且 <1.11。文件本機預覽使用 Quarto 1.10.19。後續正式的產生器更新必須保留自己的精確版本及修訂版本。表格、提示框與導覽請使用 Quarto 原生結構；來源識別資訊與產生的呈現識別資訊應分開處理。

## 文件建置 {#documentation-builds}

`scripts/pages_versions.json` 是版本目錄。`scripts/build_docs_site.py` 會固定來源修訂版本與中繼資料，在停用程式碼執行的狀態下渲染乾淨快照，並組合完整網站。目前 `main` 與 `develop` 使用各自的內容；保留的歷史版本沿用原始 README／Quarto 內文，並套用有文件記錄的呈現覆蓋層。

根目錄會選取最新穩定版本。切換版本時，若目前頁面存在就保留該頁，否則前往該版本首頁。`/main/` 與 `/develop/` 是別名。各版本網站會使用一致的行內目錄。套件版本、內容修訂版本、呈現修訂版本、Askr 修訂版本與分支，會在產生的建置記錄中分別列出。

Documentation Pages 工作流程會在推送至 `develop`／`main` 或手動派送工作流程時建置；拉取請求不會發布。必要渲染若失敗，就不會發布，並保留上一個已部署的網站。來源變更透過 `develop` 至 `main` 的升版流程推進；穩定版套件中繼資料屬於 `main`。版本目錄變更應使用既有的維護版版本工具，以及交付資料包中指派的觀察項目進行檢查。

CI 會先判定事件差異的類型。草稿拉取請求只執行分類，並刻意略過產品工作；將拉取請求標記為 Ready 後，才會啟動適用的檢查。確認為純 Markdown 文字的 `README.md` 或 `docs/**/*.md` 變更，會在本機驗證 Markdown 連結路徑；含有行內程式碼歧義的文件則走產品檢查流程。可執行或不支援的內容、QMD／Notebook 檔案、共用設定、相依套件與來源變更，也會執行產品檢查。若無法取得比較資訊，同樣執行產品檢查。

非草稿產品變更與手動派送會保留 Python 3.12／3.13 套件版本、發行套件建置、可編輯匯入與隔離 wheel 匯入檢查。獨立的 Python 3.12 `docs-smoke` 工作會使用固定版本的公開 PDK，以純檔案準備流程執行目前來源範例。這項檢查與產品測試分開，因為涵蓋範圍不同；它不是拉取請求檢查。

## 雙語文件編寫 {#bilingual-authoring}

根目錄 Quarto 專案負責完整英文版。`zh-TW/` 負責完整繁體中文版，並使用相同的頁面路徑。兩個專案各自有導覽與搜尋索引；Askr 的語言選擇器會連結至兩者首頁。更新翻譯時，請保持頁面錨點、程式碼、單位與科學意義一致。

在儲存庫根目錄執行 `quarto render --no-execute`，可渲染英文版，並由渲染後輔助工具 `scripts/build_bilingual_docs.py` 暫存共用且未修改的 Askr 擴充功能與科學素材，再渲染中文專案至 `_site/zh-TW/`。共用暫存素材是衍生副本，不是另一份可編輯的主題。若存在 `_quarto-pages.yml`，中文專案會使用 `pages` 設定檔。Notebook 儲存格與原生求解器都不會執行。請以靜態伺服器提供合併後的網站，讓各語言保有自己的導覽與搜尋索引。根專案的渲染清單須排除 `zh-TW/**`。

```bash
python3 -m http.server 4321 --bind 0.0.0.0 --directory _site
```

檢查：開啟 `/` 與 `/zh-TW/`，分別進入教學；導覽後應維持選擇的語言。

## 實作與專案參考資料 {#implementation-and-project-references}

- [架構與資料流程](docs/architecture.qmd)
- [執行環境與結果來源沿革](docs/provenance.qmd)
- [專案範圍與衍生來源](docs/goals-and-upstream.qmd)
- [專案方向](docs/roadmap.qmd)
- [歷史診斷紀錄](docs/errors.qmd)

整合時採用的衍生輸入為 gsim 下游版本 `0.2.0+scq.1`、修訂版本 `8f5dc6c05255d003a9c6d8959537bcf8068379d3`，以及歷史 SGB 修訂版本 `e74a343154c6b19b6ba32d6fb297e700cfe08ff2`。這些資料說明來源沿革，不代表執行階段相依性、支援範圍擴張或 API 相容性。目前來源幾何使用樹內的 `scgsim.geometry` API，以及用於路徑規劃與衍生建構的 `scgsim.geometry.compiler`。

## 工程收斂 {#engineering-convergence}

目前工程收斂工作仍為 `CONVERGING`。請在[收斂狀態頁](docs/convergence-status.qmd)分別追蹤語意狀態與儲存庫交付狀態。

必要交付流程是建立拉取請求、通過必要檢查，且不得略過檢查。字面名稱為 `required` 的檢查一律會產生終止結果，且只接受所選變更範圍宣告的結果。失敗或取消的工作，以及未預期的略過，都視為失敗。

每個修復拉取請求都應附上回歸案例，或說明未納入案例的理由、採用的替代證據及任何尚存缺口。若變更涉及計算，請使用獨立的公開解析回歸案例確認正規化、參與率與群組識別，並將這些結果與回傳資料相容性分開報告。計算程序不明的歷史彙總不是正確性判準；請保留其歷史紀錄，不要據此推導容許差異。

若經授權執行真實資料相容性檢查，只回報允許公開的產物雜湊、識別資訊與收據參照。請說明公開結果是否保留原始記錄（包括未知值），並回報輸入缺漏、內容不符與識別資訊不符的負向案例。公開文字不得包含私有數值、名稱、路徑或內容。若檢查尚未完成，請標記為 `IN PROGRESS`。

若 Tier A 變更受影響，請回報指派的私有基準資料狀態。若必要資料或存取權限無法取得，請記錄 `NOT RUN`，並保留相關主張待確認。經擁有者授權的執行方式，會透過公開測試入口點、明確執行旗標與 manifest 執行：

```sh
python -m pytest tests/test_private_golden.py \
  --run-private-golden \
  --private-golden-manifest /path/to/authorized-manifest.json
```

manifest 是讀取器的輸入，不代表有權存取或發布其資料。缺少必要 manifest 輸入屬於使用方式錯誤；略過的測試狀態是 `NOT RUN`，不是基準結果。公開 CI 會排除 `private_golden` 標記。基準資料必須保留在公開儲存庫之外。Tier B 仍受支援，不會因本次收斂工作而重新分類。目前範圍與待處理項目請參閱[拉取請求範本](.github/PULL_REQUEST_TEMPLATE.md)及[收斂狀態頁](docs/convergence-status.qmd)。

## 跨專案的大型提案 {#larger-proposals-across-projects}

重大架構、跨套件契約或系統層級提案，請交由最近的擁有者 Lead 處理。在撰寫執行計畫前，該 Lead 會檢查儲存庫、分支與基底修訂版本、相關來源、未結工作、契約及既有能力。請將計畫保留在目前工作資料包中。若已授權交付，儲存庫 Git 擁有者會完成指派的分支與推送步驟，並讀回遠端結果；提案不會移轉該所有權。

## 歷史結構驗收 {#historical-structural-acceptance}

v2 模組與 API 結構於 2026-10-09 獲接受，之後透過 PR34 整合。此歷史里程碑只描述結構，不會設定另一項工程收斂工作的語意或交付狀態。
