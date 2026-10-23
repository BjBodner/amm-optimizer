/** Prefix a root-relative path with the site's base path (set by SITE_BASE at build time). */
export function withBase(path: string): string {
	const base = import.meta.env.BASE_URL.replace(/\/+$/, '');
	return base + (path.startsWith('/') ? path : '/' + path);
}
