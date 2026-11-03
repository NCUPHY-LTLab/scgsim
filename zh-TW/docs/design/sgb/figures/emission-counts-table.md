| 元素 | pad_in_ground_B 中的數量 | record | 被誰引用 |
|:---|---:|:---|:---|
| vertex | 60 | `Ve` | edge |
| edge | 90 | `Ed` | wire |
| wire（迴圈） | 42 | `Wi` | face |
| face | 37（其中 1 個為 free face：port sheet） | `Fa` | shell；free face 另由 compound 引用 |
| shell | 2 | `Sh` | solid |
| solid | 2 | `So` | compound |
| compound | 1 | `Co` | 無（根節點） |
