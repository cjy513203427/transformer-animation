# Cloudflare 本地准备

目标为 Workers Static Assets，所有教学计算在浏览器执行，无 Worker 请求处理代码、账号系统或数据库。

## 配置

- `cloudflare.config.ts`：Worker 名称、兼容日期与 SPA 路由行为。
- `vite.config.ts`：仅 `--mode cloudflare` 加载 Cloudflare Vite 插件，生成纯静态 Build Output；普通开发和 `npm run build` 使用原生 Vite。
- `cf` 与 Cloudflare Vite 插件采用当前官方 beta 版本，锁文件固定解析版本。配置 API 仍可能变化，升级时需重新核对。
- Node 推荐 24.22+（24.x）；当前 CLI 声明支持 `^22.23.0 || ^24.22.0 || ^26.12.0`。较旧 Node 可能能运行前端，但不属于该 CLI 支持范围。

```powershell
npm run build:cloudflare
```

该脚本先进行类型检查，再执行 `cf build --mode cloudflare`，构建输出位于 `.cloudflare/output/v0/`。它不上传资源，也不需要 Cloudflare 凭据。常规 Vite 产物是 `dist/`。

## 发布边界

本次仅准备代码和本地构建，不登录账号、不创建云资源、不发布、不设置域名或身份策略。

配置中 `workersDev: false`、`previewUrls: false`，没有自定义域名或路由。这是尚未选择访问范围的初始状态，不等于已配置身份保护。真正发布前，要确定公开/私人/指定成员范围，并对生产、预览和自定义域名中的所有实际入口设置相应策略。私人共享可按 Cloudflare Access 的当前文档保护入口。

没有加入部署脚本或 CI 自动部署。CI 只执行检查和本地构建。获准发布后再按当时官方 CLI 文档执行，并核验真实 URL 与访问行为。

## 官方参考

- [Workers Static Assets](https://developers.cloudflare.com/workers/static-assets/)
- [Cloudflare CLI 构建与发布](https://developers.cloudflare.com/cf/projects/)
- [cloudflare.config.ts](https://developers.cloudflare.com/cf/projects/cloudflare-config/)
- [workers.dev 路由与访问范围](https://developers.cloudflare.com/workers/configuration/routing/workers-dev/)

核对日期：2026-11-05。
