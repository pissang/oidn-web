// how to run: 
// ts-node context_generator.ts ./../../src

import * as fs from 'fs';
import * as path from 'path';

// --- CONFIGURATION ---
// Directories to ignore entirely
const IGNORE_DIRS = new Set(['node_modules', '.git', 'dist', 'build', 'out', '.next', 'coverage', '.vscode']);

// File extensions to ignore (binaries, media, etc.)
const IGNORE_EXTENSIONS = new Set([
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp',
    '.mp4', '.mp3', '.wav', '.ogg',
    '.zip', '.tar', '.gz', '.7z', '.rar',
    '.pdf', '.exe', '.dll', '.so', '.dylib', '.wasm',
    '.woff', '.woff2', '.ttf', '.eot',
    '.pyc', '.class', '.jar', '.lock' // lock files are usually too large and noisy for LLMs
]);

// Maximum file size to include (default: 250KB). Prevents blowing out LLM context limits.
const MAX_FILE_SIZE_BYTES = 250 * 1024;

function buildTree(dir: string, prefix: string = ''): string {
    let treeStr = '';
    let entries;
    try {
        entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch (e) {
        return `${prefix}└── [Error reading directory]\n`;
    }

    // Filter out ignored directories
    entries = entries.filter(entry => !IGNORE_DIRS.has(entry.name));

    // Sort: directories first, then files alphabetically
    entries.sort((a, b) => {
        if (a.isDirectory() && !b.isDirectory()) return -1;
        if (!a.isDirectory() && b.isDirectory()) return 1;
        return a.name.localeCompare(b.name);
    });

    entries.forEach((entry, index) => {
        const isLast = index === entries.length - 1;
        const pointer = isLast ? '└── ' : '├── ';
        treeStr += `${prefix}${pointer}${entry.name}\n`;

        if (entry.isDirectory()) {
            const nextPrefix = prefix + (isLast ? '    ' : '│   ');
            treeStr += buildTree(path.join(dir, entry.name), nextPrefix);
        }
    });

    return treeStr;
}

function getValidFiles(dir: string): string[] {
    let results: string[] = [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
        if (IGNORE_DIRS.has(entry.name)) continue;

        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory()) {
            results = results.concat(getValidFiles(fullPath));
        } else {
            const ext = path.extname(entry.name).toLowerCase();
            if (!IGNORE_EXTENSIONS.has(ext)) {
                results.push(fullPath);
            }
        }
    }
    return results;
}

function generateLLMContext(targetPath: string) {
    const absoluteTargetPath = path.resolve(targetPath);

    if (!fs.existsSync(absoluteTargetPath) || !fs.statSync(absoluteTargetPath).isDirectory()) {
        console.error(`Error: The path "${targetPath}" is not a valid directory.`);
        process.exit(1);
    }

    const projectName = path.basename(absoluteTargetPath);
    const outputFile = path.join(process.cwd(), `${projectName}-llm-context.txt`);

    console.log(`Analyzing directory: ${absoluteTargetPath}`);

    // 1. Generate Tree
    console.log('Building directory tree...');
    let outputContent = `Project Name: ${projectName}\n`;
    outputContent += `Generated At: ${new Date().toISOString()}\n\n`;
    outputContent += `=== PROJECT STRUCTURE ===\n`;
    outputContent += `${projectName}/\n`;
    outputContent += buildTree(absoluteTargetPath);
    outputContent += `\n=== FILE CONTENTS ===\n\n`;

    // 2. Gather Files
    console.log('Reading files...');
    const files = getValidFiles(absoluteTargetPath);

    let includedCount = 0;
    let skippedCount = 0;

    for (const filePath of files) {
        const relativePath = path.relative(absoluteTargetPath, filePath);

        try {
            const stats = fs.statSync(filePath);

            if (stats.size > MAX_FILE_SIZE_BYTES) {
                outputContent += `\n---\n`;
                outputContent += `File: ${relativePath}\n`;
                outputContent += `[File skipped: Exceeds ${MAX_FILE_SIZE_BYTES / 1024}KB size limit]\n`;
                skippedCount++;
                continue;
            }

            const content = fs.readFileSync(filePath, 'utf-8');

            // Basic heuristic to skip files that were read as utf-8 but are actually binary/corrupted
            if (content.includes('\x00')) {
                skippedCount++;
                continue;
            }

            outputContent += `\n// =========================================================================\n`;
            outputContent += `// File: ${relativePath}\n`;
            outputContent += `// =========================================================================\n\n`;
            outputContent += content;
            outputContent += `\n`;
            includedCount++;

        } catch (err) {
            console.warn(`Warning: Could not read file ${relativePath}`, err);
        }
    }

    // 3. Write Output
    fs.writeFileSync(outputFile, outputContent, 'utf-8');

    console.log(`\n✅ Done!`);
    console.log(`Included ${includedCount} files (Skipped ${skippedCount} large/binary files).`);
    console.log(`Output saved to: ${outputFile}`);
}

// CLI Entry point
const args = process.argv.slice(2);
if (args.length !== 1) {
    console.log(`Usage: npx ts-node gather-context.ts <relative-path-to-project>`);
    console.log(`Example: npx ts-node gather-context.ts ../my-webgpu-engine`);
    process.exit(1);
}

generateLLMContext(args[0]);