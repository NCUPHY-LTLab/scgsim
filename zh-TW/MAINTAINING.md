<a id="maintaining-scgsim"></a>

# 維護 SCGSim

這是維護者指南：每個 CI 工作檢查什麼、文件網站如何建置與發布，以及產生的頁面如何更新。貢獻者頁面的閱讀順序，以及開 pull request 之前檢查變更的指令，寫在[為 SCGSim 貢獻](docs/contribute/index.qmd)。<a id="implementation-and-project-references"></a>

- [來源與產生的輸出](#sources-and-generated-output)
- [來源分工](#source-ownership)
- [撰寫文件](#writing-documentation)
- [原生呈現](#native-presentation)
- [文件建置](#documentation-builds)
- [雙語文件編寫](#bilingual-authoring)
- [產生的 API 參考頁](#generated-api-references)與[產生的程式碼地圖](#generated-code-map)
- [CI 工作](#ci-jobs)
- [修復用的 pull request](#fix-pull-requests)

<a id="sources-and-generated-output"></a>

## 來源與產生的輸出

`src/scgsim/` 負責可重用的產品程式碼。`docs/` 與各 README 負責撰寫的說明；`_quarto.yml` 與 `zh-TW/_quarto.yml` 負責網站導覽與執行設定。`scripts/` 負責建置與版本工具；`tests/` 負責維護中的套件測試。產生的 HTML、搜尋索引與封裝的知識套件都是衍生輸出。公開的元件來源與可執行 Notebook 屬於 OrcaPen SC PDK。

<a id="source-ownership"></a>

## 來源分工

Palace 與 AEDT 的後端實作、它們的測試與技術文件是各自獨立的來源責任，見 [AGENTS.md](../AGENTS.md)。原生求解器的觀察與結果解讀，留在各自寫明的技術界線內。協調與交付依照 [AGENTS.md](../AGENTS.md) 與共用的 `$github-flow`；重大架構、跨套件契約或系統層級的提案，也走同一個流程。<a id="larger-proposals-across-projects"></a>

<a id="writing-documentation"></a>

## 撰寫文件

SCGSim 的來源、封裝欄位、撰寫規則與術語，依照 [DOCS_STYLE.md](../DOCS_STYLE.md)。共用的 [docs-design Skill](https://github.com/arfiligol/agent-skills/blob/main/docs-design/SKILL.md) 負責讀者原則；[quarto-docs Skill](https://github.com/arfiligol/agent-skills/blob/main/quarto-docs/SKILL.md) 與它的 Askr 採用說明負責網站實作與主題使用。

<a id="native-presentation"></a>

## 原生呈現

網站使用完整的上游 Askr 擴充功能，不加本機 CSS 或 JS。Askr 的修訂版本與 Pages 建置使用的 Quarto 版本，是 `scripts/build_docs_site.py` 中的常數 `ASKR_COMMIT` 與 `QUARTO_VERSION`；`_extensions/` 中供編寫使用的副本是同一個 Askr 版本，並寫明它支援的 Quarto 範圍。表格、提示框與導覽使用原生 Quarto 結構；來源識別與產生的呈現識別分開保存。

<a id="documentation-builds"></a>

## 文件建置

`scripts/pages_versions.json` 是版本目錄。其中的 `retired_development_lines` 列出開發版根目錄已不再發布的發行線：`v<line>-dev/` 會導向目前的開發版網站，讓已安裝的該發行線預發行版所產生的連結仍能抵達頁面。開發版本號移到下一條發行線時，請把舊的發行線加進去。`scripts/build_docs_site.py` 會固定來源修訂版本與中繼資料，在停用程式碼執行的狀態下渲染乾淨的快照，並組合完整網站。目前的 `main` 與 `develop` 使用各自的內容；保留的歷史版本沿用原本的 README 與 Quarto 內文，並套用有文件記錄的呈現覆蓋層。

根目錄會選取最新的穩定版本。切換版本時，若目前頁面存在就留在該頁，否則開啟該版本的首頁。`/main/` 與 `/develop/` 是別名。每個版本網站的行內目錄都相同：發布腳本會為每個版本的每個語言專案寫入版本選單（`askr-versions.json`），渲染前寫一次、渲染後在頁面旁再寫一次；其中每個根目錄都相對於該專案的資料夾，這是 Askr 讀取它的方式。儲存庫中提交的 `askr-versions.json` 只供本機預覽使用。套件版本、內容修訂版本、呈現修訂版本、Askr 修訂版本與分支，在產生的建置紀錄中是分開的欄位。

Documentation Pages 工作流程在推送到 `develop` 與 `main`、以及手動派送時建置；pull request 不會發布。必要的渲染失敗時不會發布，上一次部署的網站保持不變。來源變更經由從 `develop` 升版進入 `main`；穩定版套件的中繼資料屬於 `main`。版本或目錄的變更，以 `tests/test_docs_site_build.py` 與 `tests/test_documentation_links.py` 檢查。

<a id="bilingual-authoring"></a>

## 雙語文件編寫

根目錄的 Quarto 專案負責英文網站；`zh-TW/` 負責繁體中文網站，兩者的頁面路徑相同。兩個專案各自有導覽與搜尋索引，Askr 的語言選擇器連結兩邊的首頁。翻譯時，請保持頁面錨點、程式碼、單位與科學意義一致。

在儲存庫根目錄執行 `quarto render --no-execute` 會渲染英文網站。它的渲染後步驟 `scripts/build_bilingual_docs.py` 會把共用的 Askr 擴充功能、`docs/assets/` 與 `docs/images/` 複製到 `zh-TW/`，把中文專案渲染到 `_site/zh-TW/`（`zh-TW/_quarto-pages.yml` 存在時使用 `pages` 設定檔），並寫出各節已移到其他頁面的轉址（`SPLIT_PAGES`）。複製過去的檔案是衍生副本，不是另一份可編輯的主題。不會執行任何 Notebook 儲存格或求解器。根專案的渲染清單排除 `zh-TW/**`。請以靜態伺服器提供合併後的網站，讓每種語言保有自己的導覽與搜尋，再開啟 `/` 與 `/zh-TW/`：

```bash
python3 -m http.server 4321 --bind 127.0.0.1 --directory _site
```

<a id="generated-api-references"></a>

## 產生的 API 參考頁

修改 `docs/reference/api/` 與 `zh-TW/docs/reference/api/` 中撰寫的使用說明 Markdown，或所屬的 Python 公開宣告，再更新並檢查衍生的參考頁：

```bash
python scripts/generate_knowledge_reference.py
python scripts/generate_knowledge_reference.py --check
```

提交的 QMD 頁面是衍生輸出。快速測試會檢查它們是否為最新；套件建置把它們收進離線知識套件，過程中不匯入 SCGSim，也不啟動求解器。

<a id="generated-pages"></a>
<a id="generated-code-map"></a>

## 產生的程式碼地圖

[程式碼地圖](docs/contribute/code-map.qmd)的階段表格（`docs/contribute/_code-map-stages.md` 與它的中文副本）由各套件資料夾的 docstring 產生。要修改某一列，請編輯該資料夾 `__init__.py` 的 docstring，在 `scripts/code_map_zh_tw.toml` 翻譯改動的文字，再執行：

```bash
python scripts/generate_code_map.py
python scripts/generate_code_map.py --check
```

各列的順序只寫在該腳本的 `AREAS` 一處，翻譯檔的每一筆都記錄它所翻譯的英文。腳本會拒絕 `AREAS` 沒有列出的套件資料夾、缺少 `In:`、`Out:` 或 `Next:` 其中一行的階段、沒有翻譯的資料夾、對應資料夾已不存在的翻譯，以及所記錄的英文已不等於 docstring 的翻譯；最後這種情況會印出需要記錄並重新翻譯的新英文。表格與 docstring 或翻譯不一致時，`tests/test_code_map.py` 會失敗。

<a id="engineering-convergence"></a>
<a id="ci-jobs"></a>

## CI 工作

變更經由 pull request 進入 `develop`，必須通過必要檢查，不得略過。每個 CI 工作都是獨立的檢查，失敗時會直接指出是哪一項：`change-scope`、`build-and-import (3.12)`、`build-and-import (3.13)`、`fast-tests`、`quality`、`docs-smoke` 與 `pdk-tests`（`.github/workflows/ci.yml`）。因條件而略過的工作（草稿 pull request，或只改文字的流程）會回報成功。分類失敗時其他工作也會略過，所以 `change-scope` 本身也是必要檢查。失敗或取消的工作都算失敗。

`change-scope` 先判定事件比較的類型。草稿 pull request 只執行這項分類；標記為 Ready 後才會啟動其他工作。只改 `README.md` 或 `docs/**/*.md` 中 Markdown 文字的變更，會在本機檢查它的 Markdown 連結路徑，而不執行產品工作。有行內程式碼歧義的文件、可執行或不支援的內容、QMD 與 Notebook 檔案、共用設定、相依套件與來源變更，都走產品工作；無法讀取比較結果時也一樣。

產品工作在非草稿 pull request、推送到 `develop` 與 `main`，以及手動派送時執行：

| 工作 | 檢查什麼 |
|---|---|
| `build-and-import (3.12)`、`(3.13)` | 在 Python 3.12 與 3.13 上檢查套件版本、發行套件建置、可編輯匯入，以及隔離環境中的 wheel 匯入。 |
| `fast-tests` | 公開的離線測試套件，但不含 `slow`、`pdk` 與 `private_golden` 測試。 |
| `quality` | `scripts/check_quality.py`：Ruff 格式與程式檢查、可讀性檢查、Pyright（標準模式，加上 `pyproject.toml` 列出的嚴格檔案），以及 `.importlinter` 的匯入層契約。 |
| `docs-smoke` | 以固定版本的公開 PDK，透過純檔案準備流程執行目前來源中的範例（`scripts/check_docs_examples.py`）；它安裝與 `aedt` extra 相同版本的 OCP，因為 AEDT 的範例會匯出 STEP。 |
| `pdk-tests` | 以 `README.md` 安裝的公開 OrcaPen SC PDK 版本，依該 PDK 凍結的鎖定檔並搭配目前的 SCGSim wheel，執行標記為 `pdk` 的測試。 |

`fast-tests` 與 `quality` 安裝 `dev` 群組，其中包含 `occt`，所以 Pyright 能解析 OCP，標記為 `occt` 的檢查也會執行。測試工作都加上 `--fail-on-skip`：它們已安裝測試需要的全部相依套件，所以任何略過都會讓執行失敗，不會無聲通過。

Nightly 工作流程（`.github/workflows/nightly.yml`）在 UTC 18:30（台灣時間 02:30）於 `develop` 執行，也可以手動派送到任何分支：

| 工作 | 檢查什麼 |
|---|---|
| `full-tests` | 完整的公開測試套件，包含 `slow` 測試。 |
| `docs-render` | `scripts/build_docs_site.py` 的 Pages 渲染，但不部署。 |
| `roster` | `scripts/roster_cases.py` 的每個版圖，以 `pdk-tests` 固定版本的 PDK 建立並切出 Surface-EPR 條帶，必須先通過 `simulation_drc`，再於預設時間預算內完成網格化；真空區域在 x 與 y 方向不加邊、上下各加 1000 µm（`scripts/check_roster.py`）。每個版圖的紀錄上傳為 `roster-records` 成品。 |

CI 只使用公開輸入，從不執行 AEDT 或 Palace。Documentation Pages 的渲染與 CI 分開：它在推送到 `develop` 與 `main` 時執行，不在 pull request 上執行。

<a id="fix-pull-requests"></a>

## 修復用的 pull request

修復用的 pull request 要附上回歸案例，或說明為什麼沒有、改用什麼證據，以及還剩下什麼缺口。若變更涉及計算，請用獨立的公開解析案例確認正規化、參與率與群組識別，並把它們和與回傳資料的相容性分開報告。計算程序不明的歷史彙總不是正確性的判準：請把它保留為歷史，不要從它推導容許差。

`private_golden` 測試讀取保存在儲存庫之外的回傳資料。它們只在帶有明確旗標與 manifest 時執行，公開 CI 會排除它們；manifest 是讀取器的輸入，不代表有權存取或公開它的資料：

```sh
python -m pytest tests/test_private_golden.py \
  --run-private-golden \
  --private-golden-manifest /path/to/authorized-manifest.json
```

這類執行的報告只列出允許公開的產物雜湊、識別資訊與收據參照，從不包含私有的數值、名稱、路徑或內容。變更的範圍與待處理項目，請見 [pull request 範本](../.github/PULL_REQUEST_TEMPLATE.md)與相關的 Issue。
