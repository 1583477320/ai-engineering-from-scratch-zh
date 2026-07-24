/**
 * 本地进度追踪器
 *
 * 所有数据存储在用户浏览器的 localStorage 中。无网络请求，无账户，无服务器。数据不会离开设备。
 *
 * 数据结构（版本化，便于后续迁移）：
 *
 *   aifs-zh:progress:v1 = {
 *     lessons: {
 *       "<课程路径>": {
 *         answers: { "<问题ID>": { picked: number, correct: boolean, t: number } },
 *         completedAt: number | null,
 *         visitedAt: number
 *       }
 *     },
 *     updatedAt: number
 *   }
 *
 * "<课程路径>" 与 data.js 中 urls 字段使用的路径一致。
 * "<问题ID>" 格式为 "<阶段>-q<索引>"，如 "pre-q0"。
 */
(function () {
  var STORAGE_KEY = 'aifs-zh:progress:v1';
  var listeners = [];

  function emptyState() {
    return { lessons: {}, updatedAt: 0 };
  }

  function read() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return emptyState();
      var parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object' || !parsed.lessons) return emptyState();
      return parsed;
    } catch (e) {
      return emptyState();
    }
  }

  function write(state) {
    state.updatedAt = Date.now();
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) {
      // 配额不足或存储被禁用，静默失败
    }
    for (var i = 0; i < listeners.length; i++) {
      try { listeners[i](state); } catch (_) {}
    }
  }

  function ensureLesson(state, path) {
    if (!state.lessons[path]) {
      state.lessons[path] = { answers: {}, completedAt: null, visitedAt: 0 };
    }
    return state.lessons[path];
  }

  function recordVisit(path) {
    if (!path) return;
    var state = read();
    var lesson = ensureLesson(state, path);
    lesson.visitedAt = Date.now();
    write(state);
  }

  function recordAnswer(path, qid, picked, correct) {
    if (!path || !qid) return;
    var state = read();
    var lesson = ensureLesson(state, path);
    lesson.answers[qid] = { picked: picked, correct: !!correct, t: Date.now() };
    write(state);
  }

  function markLessonComplete(path) {
    if (!path) return;
    var state = read();
    var lesson = ensureLesson(state, path);
    if (!lesson.completedAt) {
      lesson.completedAt = Date.now();
      write(state);
    }
  }

  function unmarkLessonComplete(path) {
    if (!path) return;
    var state = read();
    if (state.lessons[path] && state.lessons[path].completedAt) {
      state.lessons[path].completedAt = null;
      write(state);
    }
  }

  function getLessonProgress(path) {
    if (!path) return null;
    var state = read();
    return state.lessons[path] || { answers: {}, completedAt: null, visitedAt: 0 };
  }

  function isLessonComplete(path) {
    var lp = getLessonProgress(path);
    return !!(lp && lp.completedAt);
  }

  /**
   * 给定课程 URL 列表（来自 data.js），统计已完成的数量。
   * 通过匹配尾部 "lessons/.../..." 路径来匹配。
   */
  function countCompletedFromUrls(urls) {
    var state = read();
    var n = 0;
    for (var i = 0; i < urls.length; i++) {
      var path = extractPath(urls[i]);
      if (path && state.lessons[path] && state.lessons[path].completedAt) n++;
    }
    return n;
  }

  function extractPath(url) {
    if (!url) return '';
    var m = String(url).match(/(lessons\/[^/]+\/[^/]+)\/?/);
    return m ? m[1] : '';
  }

  function totalCompleted() {
    var state = read();
    var n = 0;
    for (var k in state.lessons) {
      if (state.lessons[k].completedAt) n++;
    }
    return n;
  }

  function reset() {
    try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
    for (var i = 0; i < listeners.length; i++) {
      try { listeners[i](emptyState()); } catch (_) {}
    }
  }

  function onChange(fn) {
    if (typeof fn === 'function') listeners.push(fn);
  }

  // 跨标签页同步：当用户在其他标签页清除或更新进度时，同步刷新
  window.addEventListener('storage', function (e) {
    if (e.key !== STORAGE_KEY) return;
    var state = read();
    for (var i = 0; i < listeners.length; i++) {
      try { listeners[i](state); } catch (_) {}
    }
  });

  window.AIFSProgress = {
    recordVisit: recordVisit,
    recordAnswer: recordAnswer,
    markLessonComplete: markLessonComplete,
    unmarkLessonComplete: unmarkLessonComplete,
    getLessonProgress: getLessonProgress,
    isLessonComplete: isLessonComplete,
    countCompletedFromUrls: countCompletedFromUrls,
    extractPath: extractPath,
    totalCompleted: totalCompleted,
    reset: reset,
    onChange: onChange,
  };
})();
