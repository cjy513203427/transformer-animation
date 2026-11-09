# Cloudflare 部署

目标为 Workers Static Assets，所有教学计算在浏览器执行，无 Worker 请求处理代码、账号系统或数据库。

## 发布方式：推送 main 自动部署

仓库已在 Cloudflare 控制台通过 Workers Builds 连接到 Worker `transformer-animation`。每次推送到 `main`，Cloudflare 会：

1. `npm clean-install`
2. 执行部署命令 `npx wrangler deploy`（控制台默认值，未改动）
3. wrangler 读取 [`wrangler.jsonc`](../wrangler.jsonc)，先运行 `build.command`，即 `npm run check`（类型检查、单元测试、Vite 构建），任一步失败则不部署
4. 上传 `dist/` 中的静态资源

`wrangler.jsonc` 中的 `name` 必须与控制台里连接的 Worker 名称一致，否则 Workers Builds 会报错。

本地检查部署配置（会构建，但不上传、不需要登录）：

```powershell
npx wrangler deploy --dry-run
```

## 访问范围

- 公开：`workers_dev: true`，地址为 `transformer-animation.<账号子域>.workers.dev`，任何持有链接的人都可访问。
- `preview_urls: false`：不生成版本预览 URL。
- 没有自定义域名或路由。
- 改为私人访问时，在控制台 Worker → Settings → Domains & Routes → workers.dev 启用 Cloudflare Access；若以后添加自定义域名或预览 URL，也要分别加上相同的保护。

## `cf` CLI 构建（暂不用于发布）

- `cloudflare.config.ts`：供新的 `cf` CLI 使用，名称与 URL 设置需和 `wrangler.jsonc` 保持一致。
- `vite.config.ts`：仅 `--mode cloudflare` 加载 Cloudflare Vite 插件，生成 `.cloudflare/output/v0/`；普通开发、`npm run build` 和 wrangler 部署都使用原生 Vite，输出 `dist/`。
- `cf` 与 Cloudflare Vite 插件采用官方 beta 版本，配置 API 仍可能变化。

```powershell
npm run build:cloudflare
```

GitHub Actions（`.github/workflows/check.yml`）只做检查和构建，不部署。

## 官方参考

- [Workers Static Assets](https://developers.cloudflare.com/workers/static-assets/)
- [Workers Builds](https://developers.cloudflare.com/workers/ci-cd/builds/)
- [Wrangler 配置](https://developers.cloudflare.com/workers/wrangler/configuration/)
- [workers.dev 路由与访问范围](https://developers.cloudflare.com/workers/configuration/routing/workers-dev/)
- [Cloudflare CLI 构建与发布](https://developers.cloudflare.com/cf/projects/)

核对日期：2026-11-09。
