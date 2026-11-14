# 維護 SCGSim {#maintaining-scgsim}

本參考頁說明文件建置與套件維護方式。次要的[維護者指南](docs/ownership.qmd)連結至實作與專案參考資料。研究者無須上層工作區，可從[課程](docs/examples.qmd)開始。

## 來源與產生輸出 {#sources-and-generated-output}

`src/scgsim/` 負責可重用產品程式碼。`docs/` 與 README 負責撰寫說明；`_quarto.yml` 負責網站導覽與執行設定。`scripts/` 負責建置及版本工具；`tests/` 負責維護中的套件測試。產生的 HTML、搜尋索引與封裝知識都是衍生輸出。公開元件與可執行 Notebook 來源由 OrPen SC PDK 維護。

## 來源分工 {#source-ownership}

Palace 與 AEDT 後端實作、測試及技術文件仍屬不同來源責任，詳見[儲存庫根目錄指南](../AGENTS.md)。原生求解器觀察與結果解讀應維持各自明定的技術界線。貢獻者協調與交付請遵循根目錄指南及共用 `$github-flow`。

## 原生呈現 {#native-presentation}

網站使用完整的上游 Askr 擴充功能，搭配原生 Quarto 結構。編輯用擴充功能與多版本網站建置器固定使用 Askr v0.14.0，修訂版本為 `ab12c10b61d590c00ce1316f700ed56227027d71`，支援 Quarto >=1.9.38 且 <1.11。本機文件檢查使用 Quarto 1.10.19；Pages 工作流程使用相容範圍內的 1.10.18。

表格、提示框與導覽採用原生元件；程式碼以 `code-overflow: scroll` 保留原始行結構。來源修訂版本與呈現修訂版本分別記錄。

## 文件建置 {#documentation-builds}

`scripts/pages_versions.json` 是版本目錄。其中的 `retired_development_lines` 列出開發版根目錄已不再發布的發行線：`v<line>-dev/` 會導向目前的開發版網站，讓已安裝的該發行線預發行版所產生的連結仍能抵達頁面。開發版本號移到下一條發行線時，請把舊的發行線加進去。`scripts/build_docs_site.py` 會固定來源修訂版本與中繼資料，在停用程式碼執行的狀態下渲染乾淨快照，並組合完整網站。目前 `main` 與 `develop` 使用各自的內容；保留的歷史版本沿用原始 README／Quarto 內文，並套用有文件記錄的呈現覆蓋層。

根目錄會選取最新穩定版本。切換版本時，若目前頁面存在就保留該頁，否則前往該版本首頁。`/main/` 與 `/develop/` 是別名。各版本網站會使用一致的行內目錄。套件版本、內容修訂版本、呈現修訂版本、Askr 修訂版本與分支，會在產生的建置記錄中分別列出。

Documentation Pages 工作流程會在推送至 `develop`／`main` 或手動派送工作流程時建置；拉取請求不會發布。必要渲染若失敗，就不會發布，並保留上一個已部署的網站。來源變更透過 `develop` 至 `main` 的升版流程推進；穩定版套件中繼資料屬於 `main`。版本目錄變更應使用既有的維護版版本工具，以及交付資料包中指派的觀察項目進行檢查。

CI 會先判定事件差異的類型。草稿拉取請求只執行分類，並刻意略過產品工作；將拉取請求標記為 Ready 後，才會啟動適用的檢查。確認為純 Markdown 文字的 `README.md` 或 `docs/**/*.md` 變更，會在本機驗證 Markdown 連結路徑；含有行內程式碼歧義的文件則走產品檢查流程。可執行或不支援的內容、QMD／Notebook 檔案、共用設定、相依套件與來源變更，也會執行產品檢查。若無法取得比較資訊，同樣執行產品檢查。

非草稿產品變更、推送至 `develop` 或 `main`，以及手動派送，會執行快速層。`build-and-import` 在 Python 3.12 與 3.13 檢查套件版本、發行套件建置、可編輯匯入與隔離 wheel 匯入。`fast-tests` 執行公開離線測試套件，但不含 `slow`、`pdk` 與 `private_golden` 測試。`quality` 執行完整品質入口 `scripts/check_quality.py`，涵蓋 Ruff 格式與程式檢查、可讀性、標準及原有嚴格 Pyright 設定，以及六條匯入層契約。兩者都安裝 `occt` 相依群組，所以 Pyright 能解析 OCP，標記為 `occt` 的 OCCT 檢查也會執行。`docs-smoke` 使用固定版本的公開 PDK，以純檔案準備流程執行目前來源範例。`pdk-tests` 以 `README.md` 安裝的公開 OrcaPen SC PDK 修訂版本，依該 PDK 凍結的鎖定檔建立環境並安裝目前的 SCGSim wheel，執行標記為 `pdk` 的測試。測試工作都會加上 `--fail-on-skip`：它們已安裝測試所需的全部相依套件，因此任何略過都會使執行失敗，不會無聲通過。

Nightly 工作流程在 UTC 18:30（台灣時間 02:30）於 `develop` 執行重型層，也可手動派送到任何分支。`full-tests` 執行含 `slow` 測試的完整公開套件，`docs-render` 執行 Pages 渲染但不部署。`roster` 以 `pdk-tests` 固定的 PDK 版本建立定義「Palace 穩定」的版圖（issue 141），切出 Surface-EPR 條帶，並要求每個版圖都通過 `simulation_drc`，且在 x、y 方向不加、z 方向各加 1000 µm 的真空區中於預設時間預算內完成網格化（`scripts/check_roster.py`）；每個版圖的紀錄會上傳為 `roster-records` 成品。任何失敗都會使該次執行失敗。CI 只使用公開輸入並排除私有基準資料，不會呼叫 AEDT 或 Palace 求解器。

## 產生 API 參考頁 {#generated-api-references}

修改 `docs/api/` 與 `zh-TW/docs/api/` 的正式 Markdown 使用說明，或所屬 Python 公開宣告，再產生並檢查衍生頁面：

```bash
python scripts/generate_knowledge_reference.py
python scripts/generate_knowledge_reference.py --check
```

提交的 QMD 是衍生輸出。一般測試會檢查內容是否更新；套件建置將它們收錄至離線知識套件，過程不匯入 SCGSim 或啟動求解器。

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

整合時採用的衍生輸入為 gsim 下游版本 `0.2.0+scq.1`、修訂版本 `8f5dc6c05255d003a9c6d8959537bcf8068379d3`，以及歷史 SGB 修訂版本 `e74a343154c6b19b6ba32d6fb297e700cfe08ff2`。這些資料說明來源沿革，不代表執行階段相依性、支援範圍擴張或 API 相容性。目前來源幾何使用樹內的 `scgsim.geometry` API。Palace 以 `scgsim.geometry.sgb` 建立幾何，並用 `scgsim.meshing` 產生網格。SGB 位於支援的入口（`GeometryPlan` 與模擬類別）以下：以自行建立的輸入直接呼叫時，先決條件是該入口所保證的事項，見[支援範圍](docs/supported-scope.qmd#supported-entry)。

## 工程收斂 {#engineering-convergence}

請在各項工作的 GitHub Issue 與 pull request 中，分別追蹤語意狀態與儲存庫交付狀態。

必要交付流程是建立拉取請求、通過必要檢查，且不得略過檢查。每個 CI 工作都是獨立的檢查，失敗時會直接顯示是哪一項：`change-scope`、`build-and-import (3.12)`、`build-and-import (3.13)`、`fast-tests`、`quality`、`docs-smoke` 與 `pdk-tests`。因條件而略過的工作（草稿拉取請求或純文件流程）會回報成功。分類失敗時其他工作也會略過，所以 `change-scope` 本身也是必要檢查。失敗或取消的工作都視為失敗。

每個修復拉取請求都應附上回歸案例，或說明未納入案例的理由、採用的替代證據及任何尚存缺口。若變更涉及計算，請使用獨立的公開解析回歸案例確認正規化、參與率與群組識別，並將這些結果與回傳資料相容性分開報告。計算程序不明的歷史彙總不是正確性判準；請保留其歷史紀錄，不要據此推導容許差異。

若經授權執行真實資料相容性檢查，只回報允許公開的產物雜湊、識別資訊與收據參照。請說明公開結果是否保留原始記錄（包括未知值），並回報輸入缺漏、內容不符與識別資訊不符的負向案例。公開文字不得包含私有數值、名稱、路徑或內容。若檢查尚未完成，請標記為 `IN PROGRESS`。

若 Tier A 變更受影響，請回報指派的私有基準資料狀態。若必要資料或存取權限無法取得，請記錄 `NOT RUN`，並保留相關主張待確認。經擁有者授權的執行方式，會透過公開測試入口點、明確執行旗標與 manifest 執行：

```sh
python -m pytest tests/test_private_golden.py \
  --run-private-golden \
  --private-golden-manifest /path/to/authorized-manifest.json
```

manifest 是讀取器的輸入，不代表有權存取或發布其資料。缺少必要 manifest 輸入屬於使用方式錯誤；略過的測試狀態是 `NOT RUN`，不是基準結果。公開 CI 會排除 `private_golden` 標記。基準資料必須保留在公開儲存庫之外。Tier B 仍受支援，不會因本次收斂工作而重新分類。目前範圍與待處理項目請參閱[拉取請求範本](../.github/PULL_REQUEST_TEMPLATE.md)與相關的 Issue。

## 跨專案的大型提案 {#larger-proposals-across-projects}

重大架構、跨套件契約或系統層級提案，請遵循儲存庫共用的貢獻流程。並維持 [AGENTS.md](../AGENTS.md) 所列的來源與求解器所有權界線。

## 歷史結構驗收 {#historical-structural-acceptance}

v2 模組與 API 結構於 2026-10-09 獲接受，之後透過 PR34 整合。此歷史里程碑只描述結構，不會設定另一項工程收斂工作的語意或交付狀態。
