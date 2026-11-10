# Cloudflare 部署

目标为 Workers Static Assets，所有教学计算在浏览器执行，无 Worker 请求处理代码、账号系统或数据库。

## 发布方式：推送 main 自动部署

仓库已在 Cloudflare 控制台通过 Workers Builds 连接到 Worker `transformer-animation`。每次推送到 `main`，Cloudflare 会：

1. `npm clean-install`
2. 执行部署命令 `npx wrangler deploy`（控制台默认值，未改动）
3. wrangler 读取 [`wrangler.jsonc`](../wrangler.jsonc)，先运行 `build.command`，即 `npm run build:deploy`（类型检查、单元测试、带访问统计脚本的 Vite 构建），任一步失败则不部署
4. 上传 `dist/` 中的静态资源

`wrangler.jsonc` 中的 `name` 必须与控制台里连接的 Worker 名称一致，否则 Workers Builds 会报错。

本地检查部署配置（会构建，但不上传、不需要登录）：

```powershell
npm run deploy -- --dry-run
```

手动发布到同一个 Worker：

```powershell
npm run deploy
```

这两个命令都会先运行 `predeploy`，仅清理项目内生成的 `.cloudflare/output/`，再调用 Wrangler。清理可重复执行，不会删除 `cloudflare.config.ts`、`wrangler.jsonc` 或源代码。

如果之前运行过 `npm run build:cloudflare`，直接执行 `npx wrangler deploy` 会因检测到 Cloudflare Build Output 而拒绝发布。使用上面的 npm 命令即可清理旧产物并重新构建 `dist/`。本项目生产发布仍使用 `wrangler.jsonc`；不要仅因为报错提示就改用 `cf deploy`，后者使用 `cloudflare.config.ts` 的另一套构建与配置。

## 访问范围

- 公开：`workers_dev: true`，地址为 `transformer-animation.<账号子域>.workers.dev`，任何持有链接的人都可访问。
- `preview_urls: false`：不生成版本预览 URL。
- 没有自定义域名或路由。
- 英文、德文页面是同一个 Worker 下的 `/en/…`、`/de/…` 路径，由 `not_found_handling: single-page-application` 回退到 `index.html`，不需要额外配置；访问范围与中文页面相同。
- 改为私人访问时，在控制台 Worker → Settings → Domains & Routes → workers.dev 启用 Cloudflare Access；若以后添加自定义域名或预览 URL，也要分别加上相同的保护。

## 访问统计

线上站点使用 Cloudflare Web Analytics（控制台 → Web Analytics，站点 `transformer-animation.cjyhobbyprojects.workers.dev`）。站点没有接入 Cloudflare 代理的自有域名，不能自动注入，因此手动加入 JS 代码：

- 代码与 token 写在 `vite.config.ts`。token 本来就会出现在页面源码中，不是秘密。
- 只有 `vite build --mode deploy`（即 `npm run build:deploy`）会把脚本插入 `index.html`；`npm run dev`、`npm run build`、`npm run preview` 与 e2e 测试都不加载，避免本地访问计入统计。两种构建的 JS/CSS 资源相同。
- 站内导航是普通链接，每次换页都会完整加载页面；实验参数写在 URL hash 中，不算新的页面浏览。
- 三种语言的路径不同（`/moe`、`/en/moe`、`/de/moe`），在统计里按路径分开显示，可以看出各语言的访问量。
- Workers & Pages → Worker → Metrics 只有原始请求数，每个 JS、CSS、字体文件都算一次，爬虫也计入，不区分页面和访客。

## `cf` CLI 构建（暂不用于发布）

- `cloudflare.config.ts`：供新的 `cf` CLI 使用，名称与 URL 设置需和 `wrangler.jsonc` 保持一致。
- `vite.config.ts`：仅 `--mode cloudflare` 加载 Cloudflare Vite 插件，生成 `.cloudflare/output/v0/`；普通开发、`npm run build` 和 wrangler 部署（`--mode deploy`）都不加载该插件，输出 `dist/`。
- `cf` 与 Cloudflare Vite 插件采用官方 beta 版本，配置 API 仍可能变化。
- 这项构建会留下 `.cloudflare/output/`；之后本地发布请使用 `npm run deploy`，由 `predeploy` 清理该目录。

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

核对日期：2026-11-10。
