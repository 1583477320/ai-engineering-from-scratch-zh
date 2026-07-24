#!/usr/bin/env node
/**
 * AI Engineering 从零 - 站点构建脚本
 *
 * 扫描 lessons/ 目录，提取元数据，生成 site/data.js
 *
 * 用法：node site/build.js
 */

const fs = require("fs");
const path = require("path");

// ── 项目根目录 ──────────────────────────────────────────────
const ROOT = path.resolve(__dirname, "..");
const LESSONS_DIR = path.join(ROOT, "lessons");
const OUTPUT_FILE = path.join(__dirname, "data.js");

// ── 工具函数 ────────────────────────────────────────────────

/**
 * 统计目录下所有文件的代码行数
 */
function countLines(dirPath, extensions) {
  let total = 0;
  if (!fs.existsSync(dirPath)) return total;
  for (const entry of fs.readdirSync(dirPath)) {
    const full = path.join(dirPath, entry);
    const stat = fs.statSync(full);
    if (stat.isFile()) {
      const ext = path.extname(entry).toLowerCase();
      if (extensions.includes(ext)) {
        const content = fs.readFileSync(full, "utf-8");
        total += content.split("\n").length;
      }
    } else if (stat.isDirectory()) {
      total += countLines(full, extensions);
    }
  }
  return total;
}

/**
 * 统计 Markdown 行数
 */
function countMdLines(dirPath) {
  return countLines(dirPath, [".md"]);
}

/**
 * 从 docs/index.md 提取元数据
 */
function parseMetadata(mdPath) {
  if (!fs.existsSync(mdPath)) return null;
  const content = fs.readFileSync(mdPath, "utf-8");
  if (!content.trim()) return null;

  const lines = content.split("\n");

  // ── 课程名：第一个 # 标题 ──
  let name = "";
  for (const line of lines) {
    const m = line.match(/^#\s+(.+)/);
    if (m) {
      name = m[1].trim();
      break;
    }
  }

  // ── 摘要：第一个 > 引用 ──
  let summary = "";
  for (const line of lines) {
    const m = line.match(/^>\s*(.+)/);
    if (m) {
      summary = m[1].trim();
      break;
    }
  }

  // ── 元数据字段 ──
  let type = "";
  let lang = "";
  let prerequisites = "";
  let time = "";
  let tier = "";
  let courseLinks = "";

  for (const line of lines) {
    let m;

    // 类型：概念课 / 实现课 / 构建
    m = line.match(/^\*\*类型[：:]\*\*\s*(.+)/);
    if (m) { type = m[1].trim(); continue; }

    // 编程语言 或 语言
    m = line.match(/^\*\*(?:编程)?语言[：:]\*\*\s*(.+)/);
    if (m) { lang = m[1].trim(); continue; }

    // 前置知识
    m = line.match(/^\*\*前置知识[：:]\*\*\s*(.+)/);
    if (m) { prerequisites = m[1].trim(); continue; }

    // 预计时间
    m = line.match(/^\*\*预计时间[：:]\*\*\s*(.+)/);
    if (m) { time = m[1].trim(); continue; }

    // 所处阶段：Tier 1 / Tier 2 / Tier 3
    m = line.match(/^\*\*所处阶段[：:]\*\*\s*(.+)/);
    if (m) { tier = m[1].trim(); continue; }

    // 关联课程
    m = line.match(/^\*\*关联课程[：:]\*\*\s*(.+)/);
    if (m) { courseLinks = m[1].trim(); continue; }
  }

  // ── 关键词：所有 ### 标题 ──
  const keywords = [];
  for (const line of lines) {
    const m = line.match(/^###\s+(.+)/);
    if (m) {
      keywords.push(m[1].trim());
    }
  }

  // ── 总行数（用于状态判断） ──
  const totalLines = lines.length;

  return {
    name,
    summary,
    type,
    lang,
    prerequisites,
    time,
    tier,
    courseLinks,
    keywords,
    totalLines,
  };
}

/**
 * 判断课程状态
 */
function determineStatus(mdPath) {
  if (!fs.existsSync(mdPath)) return "planned";
  const stat = fs.statSync(mdPath);
  if (stat.size === 0) return "planned";

  const content = fs.readFileSync(mdPath, "utf-8");
  const lineCount = content.split("\n").length;

  if (lineCount >= 100) return "complete";
  return "in-progress";
}

/**
 * 从目录名提取序号和名称
 * "01-开发环境配置" → { num: 1, name: "开发环境配置" }
 */
function parseDirName(dirname) {
  const m = dirname.match(/^(\d+)[-_](.+)/);
  if (!m) return null;
  return { num: parseInt(m[1], 10), name: m[2] };
}

// ── 主构建逻辑 ──────────────────────────────────────────────

function build() {
  // 读取所有阶段目录
  const phaseDirs = fs
    .readdirSync(LESSONS_DIR)
    .filter((d) => {
      const full = path.join(LESSONS_DIR, d);
      return fs.statSync(full).isDirectory();
    })
    .sort();

  const phases = [];

  for (const phaseDir of phaseDirs) {
    const parsed = parseDirName(phaseDir);
    if (!parsed) {
      console.warn(`[跳过] 无法解析阶段目录: ${phaseDir}`);
      continue;
    }

    const phasePath = path.join(LESSONS_DIR, phaseDir);
    const lessons = [];

    // 读取阶段下的所有课程目录
    const lessonDirs = fs
      .readdirSync(phasePath)
      .filter((d) => {
        const full = path.join(phasePath, d);
        return fs.statSync(full).isDirectory();
      })
      .sort();

    for (const lessonDir of lessonDirs) {
      const lessonParsed = parseDirName(lessonDir);
      if (!lessonParsed) {
        console.warn(`  [跳过] 无法解析课程目录: ${phaseDir}/${lessonDir}`);
        continue;
      }

      const lessonPath = path.join(phasePath, lessonDir);
      const mdPath = path.join(lessonPath, "docs", "index.md");
      const quizPath = path.join(lessonPath, "quiz.json");
      const codePath = path.join(lessonPath, "code");

      // 提取元数据
      const meta = parseMetadata(mdPath);
      const status = determineStatus(mdPath);

      // 计算代码和文档行数
      const codeLines = countLines(codePath, [
        ".py", ".js", ".ts", ".rs", ".go", ".java",
        ".ipynb", ".sh", ".yaml", ".yml", ".json", ".toml",
      ]);
      const docLines = countMdLines(path.join(lessonPath, "docs"));

      // 相对路径（相对于项目根目录）
      const relPath = path.join("lessons", phaseDir, lessonDir);

      lessons.push({
        lessonNum: lessonParsed.num,
        name: meta ? meta.name : lessonParsed.name,
        status,
        type: meta ? meta.type : "",
        lang: meta ? meta.lang : "",
        prerequisites: meta ? meta.prerequisites : "",
        time: meta ? meta.time : "",
        tier: meta ? meta.tier : "",
        courseLinks: meta ? meta.courseLinks : "",
        path: relPath,
        summary: meta ? meta.summary : "",
        keywords: meta ? meta.keywords : [],
        codeLines,
        docLines,
        hasCode: codeLines > 0,
        hasQuiz: fs.existsSync(quizPath),
      });
    }

    // 计算阶段状态：所有课程都 complete 才算 complete
    const completedCount = lessons.filter((l) => l.status === "complete").length;
    const phaseStatus =
      lessons.length > 0 && completedCount === lessons.length
        ? "complete"
        : completedCount > 0
          ? "in-progress"
          : "planned";

    phases.push({
      id: parsed.num,
      name: parsed.name,
      status: phaseStatus,
      completedLessons: completedCount,
      totalLessons: lessons.length,
      lessons,
    });
  }

  // ── 汇总统计 ──
  const totalLessons = phases.reduce((sum, p) => sum + p.totalLessons, 0);
  const totalCompleted = phases.reduce((sum, p) => sum + p.completedLessons, 0);

  // ── 生成 data.js ──
  const output = generateDataJs(phases, totalLessons, totalCompleted);
  fs.writeFileSync(OUTPUT_FILE, output, "utf-8");

  console.log(`\n构建完成！`);
  console.log(`  阶段数: ${phases.length}`);
  console.log(`  总课程: ${totalLessons}`);
  console.log(`  已完成: ${totalCompleted}`);
  console.log(`  进行中: ${phases.reduce((s, p) => s + p.lessons.filter((l) => l.status === "in-progress").length, 0)}`);
  console.log(`  未开始: ${phases.reduce((s, p) => s + p.lessons.filter((l) => l.status === "planned").length, 0)}`);
  console.log(`\n输出文件: ${path.relative(ROOT, OUTPUT_FILE)}`);
}

/**
 * 生成 data.js 内容
 */
function generateDataJs(phases, totalLessons, totalCompleted) {
  const parts = [];

  parts.push(`/**`);
  parts.push(` * AI Engineering 从零 - 课程数据`);
  parts.push(` * `);
  parts.push(` * 由 site/build.js 自动生成，请勿手动编辑`);
  parts.push(` * 生成时间: ${new Date().toISOString()}`);
  parts.push(` */`);
  parts.push(``);
  parts.push(`const TOTAL_LESSONS = ${totalLessons};`);
  parts.push(`const TOTAL_COMPLETED = ${totalCompleted};`);
  parts.push(``);
  parts.push(`const PHASES = [`);

  for (let i = 0; i < phases.length; i++) {
    const phase = phases[i];
    parts.push(`  {`);
    parts.push(`    id: ${phase.id}, name: "${escapeJs(phase.name)}", status: "${phase.status}",`);
    parts.push(`    completedLessons: ${phase.completedLessons}, totalLessons: ${phase.totalLessons},`);
    parts.push(`    lessons: [`);

    for (const lesson of phase.lessons) {
      parts.push(`      {`);
      parts.push(`        lessonNum: ${lesson.lessonNum}, name: "${escapeJs(lesson.name)}", status: "${lesson.status}",`);
      parts.push(`        type: "${escapeJs(lesson.type)}", lang: "${escapeJs(lesson.lang)}",`);
      parts.push(`        prerequisites: "${escapeJs(lesson.prerequisites)}", time: "${escapeJs(lesson.time)}", tier: "${escapeJs(lesson.tier)}",`);
      parts.push(`        courseLinks: "${escapeJs(lesson.courseLinks)}",`);
      parts.push(`        path: "${escapeJs(lesson.path)}",`);
      parts.push(`        summary: "${escapeJs(lesson.summary)}",`);
      parts.push(`        keywords: [${lesson.keywords.map((k) => `"${escapeJs(k)}"`).join(", ")}],`);
      parts.push(`        codeLines: ${lesson.codeLines}, docLines: ${lesson.docLines}, hasCode: ${lesson.hasCode}, hasQuiz: ${lesson.hasQuiz}`);
      parts.push(`      },`);
    }

    parts.push(`    ]`);
    parts.push(`  }${i < phases.length - 1 ? "," : ""}`);
  }

  parts.push(`];`);
  parts.push(``);

  return parts.join("\n");
}

/**
 * 转义 JavaScript 字符串中的特殊字符
 */
function escapeJs(str) {
  if (!str) return "";
  return str
    .replace(/\\/g, "\\\\")
    .replace(/"/g, '\\"')
    .replace(/\n/g, "\\n")
    .replace(/\r/g, "\\r");
}

// ── 执行 ──
build();
