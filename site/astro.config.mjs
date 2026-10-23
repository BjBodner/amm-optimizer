// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import ammGuards from './integrations/amm-guards.mjs';
import baseLinks from './integrations/base-links.mjs';

const launch = process.env.AMM_LAUNCH === '1';
const showAuthor = process.env.AMM_SHOW_AUTHOR === '1';
const siteBase = process.env.SITE_BASE || '/'; // e.g. /amm-optimizer for a GitHub project site
const siteUrl = process.env.SITE_URL; // set for the public build, e.g. https://example.github.io

const webSite = {
	'@context': 'https://schema.org',
	'@type': 'WebSite',
	name: 'AMM (Adaptive Multi Mode) optimizer',
	alternateName: ['AMM optimizer', 'AMM'],
};

export default defineConfig({
	site: launch ? siteUrl : undefined,
	base: siteBase,
	integrations: [
		starlight({
			title: 'AMM',
			description:
				'AMM (Adaptive Multi Mode) is a derivative-free optimizer that reads your function with cheap probes, then routes the evaluation budget to the right classical algorithm.',
			customCss: ['./src/styles/custom.css'],
			// A public build is indexable; the private preview is not.
			head: [
				...(launch ? [] : [{ tag: 'meta', attrs: { name: 'robots', content: 'noindex, nofollow' } }]),
				{ tag: 'script', attrs: { type: 'application/ld+json' }, content: JSON.stringify(webSite) },
			],
			social: launch
				? [{ icon: 'github', label: 'GitHub', href: process.env.GITHUB_URL ?? 'https://github.com/' }]
				: [],
			lastUpdated: false,
			pagination: false,
			sidebar: [
				{
					label: 'Start',
					items: [
						{ label: 'Get started', slug: 'get-started' },
						{ label: 'How it works', slug: 'how-it-works' },
						{ label: 'Guides', slug: 'guides' },
					],
				},
				{
					label: 'Evidence',
					items: [
						{ label: 'Results', slug: 'results' },
						{ label: 'Limits and honesty', slug: 'limits' },
						{ label: 'Reproduce', slug: 'reproduce' },
						{ label: 'Research log', slug: 'research-log' },
					],
				},
				{
					label: 'Paper',
					items: [{ label: 'Paper and cite', slug: 'paper' }],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'API reference', slug: 'api' },
						{ label: 'Changelog', slug: 'changelog' },
						{ label: 'FAQ', slug: 'faq' },
					],
				},
				{
					label: 'More',
					items: [
						{ label: 'Landscape Atlas', slug: 'atlas' },
						{ label: 'Writing', slug: 'writing' },
						{ label: 'About', slug: 'about' },
					],
				},
			],
		}),
		baseLinks({ base: siteBase }),
		ammGuards({ launch, showAuthor }),
	],
});
