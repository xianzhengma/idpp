# Oracle JSON（权威 test 列表）

公开数据 **test1 / test2** 的样本集合以 Genesis Oracle 导出的 JSON 为准。

## 使用方式

1. 在集群上保证以下目录可读（或设置环境变量覆盖）：

   - 默认：`/work/guanqi/Genesis/oracle_py_files/`
   - 覆盖：`export ORACLE_JSON_DIR=/path/to/dir`

2. 该目录下需包含与 [`oracle_lane_manifest.yaml`](../oracle_lane_manifest.yaml) 中 `oracle_json` 字段同名的文件，例如 `elasticity_absolute_test1.json`。

3. 若需在本仓库内做 **CI / 无 Genesis 路径** 的校验，可将上述 JSON **复制**到本目录（`oracle_lists/*.json`），并设置：

   ```bash
   export ORACLE_JSON_DIR=/work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/data/oracle_lists
   ```

   **不要**将含密钥或其它敏感内容的文件放入此目录；仅列表 JSON。

## JSON 结构（脚本兼容）

- **字符串列表**：每项为 basename（无 `_rgb.mp4` 后缀）或带 `_rgb.mp4` 的文件名。
- **列表的列表 / 元组**：取前 1～2 个元素转为字符串 basename（与 `webvid.py` 中 `video_list_json` 解析一致）。
- **对象（dict）**：若顶层为 dict 且 key 为 basename（常见于 `bounce_analysis_results` 风格），则允许 key 集合作为 oracle（不含 `_rgb.mp4` 后缀）。
