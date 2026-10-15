import { mkdir, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const outputDir = join(process.cwd(), 'public', 'images');
const candidates = [
  {
    url: 'https://zefirka.club/uploads/posts/2022-12/1672261699_75-zefirka-club-p-viuchit-tekhniki-iz-naruto-rasengan-87.png',
    filename: 'rasengan-reference.png',
    referer: 'https://zefirka.club/',
  },
  {
    url: 'https://www.looper.com/img/gallery/the-most-important-jutsu-that-naruto-ever-learned/the-rasengan-is-no-slouch-either-1655876917.jpg',
    filename: 'rasengan-reference.jpg',
    referer: 'https://www.looper.com/903145/the-most-important-jutsu-that-naruto-ever-learned/',
  },
  {
    url: 'https://wallpaper.forfun.com/fetch/f4/f4bc7b8352674e59dbee35fbfb0dcfb8.jpeg',
    filename: 'rasengan-reference.jpeg',
    referer: 'https://wallpaper.mob.gr.jp/pc/image/anime-boruto_naruto_the_movie-boruto_uzumaki-rasengan_naruto-408200.html',
  },
];

await mkdir(outputDir, { recursive: true });
for (const candidate of candidates) {
  await unlink(join(outputDir, candidate.filename)).catch(() => undefined);
}

let saved = false;
for (const candidate of candidates) {
  try {
    const response = await fetch(candidate.url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; KonohaRasenganLab/1.0)',
        Referer: candidate.referer,
        Accept: 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
      },
      signal: AbortSignal.timeout(8000),
    });
    if (!response.ok) {
      console.warn('Rasengan image source returned', response.status, candidate.url);
      continue;
    }
    const contentType = response.headers.get('content-type') ?? '';
    if (!contentType.toLowerCase().startsWith('image/')) {
      console.warn('Rasengan image source did not return an image:', candidate.url);
      continue;
    }
    const bytes = Buffer.from(await response.arrayBuffer());
    if (bytes.byteLength < 10_000) {
      console.warn('Rasengan image source returned an unexpectedly small file:', candidate.url);
      continue;
    }
    await writeFile(join(outputDir, candidate.filename), bytes);
    console.log('Cached Rasengan technique-card image:', candidate.filename, bytes.byteLength, 'bytes');
    saved = true;
    break;
  } catch (error) {
    console.warn('Could not cache image from', candidate.url, error instanceof Error ? error.message : String(error));
  }
}

if (!saved) {
  // Do not fail builds over an optional editorial image; the app has browser-side fallbacks.
  console.warn('No image host was reachable during this build; runtime image fallbacks remain enabled.');
}
