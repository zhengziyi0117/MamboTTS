`mambo-sans.woff2` 是 Noto Sans SC 的 WOFF2 子集，包含拉丁字符、CJK 汉字、假名及常用标点。
用于没有安装完整中文字体的浏览器，页面不需要访问在线字体服务。

原字体：[Google Fonts / Noto Sans SC](https://github.com/google/fonts/tree/main/ofl/notosanssc)。
许可及版权声明见 [OFL.txt](OFL.txt)。

使用 FontTools 与 Brotli，从上游 `NotoSansSC[wght].ttf` 生成：

```bash
pyftsubset 'NotoSansSC[wght].ttf' --output-file=mambo-sans.woff2 --flavor=woff2 \
  --unicodes='U+0000-00FF,U+2000-206F,U+3000-30FF,U+3400-9FFF,U+FF00-FFEF' \
  --layout-features='*'
```
