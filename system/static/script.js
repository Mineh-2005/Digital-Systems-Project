/* MarketMatch — Shared Utilities */

/* Auth */

/**
 * Checks if user is logged in, redirects to login if not.
 * Call at the top of window.onload on protected pages.
 */
function checkLogin() {
    const isLoggedIn = localStorage.getItem('userLoggedIn');
    if (!isLoggedIn) {
        window.location.href = '/login';
        return false;
    }
    return true;
}

/**
 * Logs the user out — clears all localStorage and redirects to login.
 */
function logout() {
    localStorage.clear();
    window.location.href = '/login';
}

/* Nav */

/**
 * Initialises the shared nav:
 * - Sets the Welcome, [Name]! greeting
 * - Adds scroll shadow to nav
 *
 * @param {string} greetingId - ID of the greeting <span> element
 * @param {string} navId      - ID of the <nav> element (default: 'mmNav')
 */
function initNav(greetingId = 'navGreeting', navId = 'mmNav') {
    const name = localStorage.getItem('userName') || '';
    const isAdmin = localStorage.getItem('isAdmin') === 'true';

    const greetingEl = document.getElementById(greetingId);
    if (greetingEl) {
        greetingEl.textContent = name ? 'Welcome, ' + name + '!' : 'Welcome!';
        greetingEl.classList.add('ready');
    }

    const adminLink = document.getElementById('adminNavLink');
    if (adminLink) {
        adminLink.style.display = isAdmin ? 'inline-flex' : 'none';
    }

    const nav = document.getElementById(navId);
    if (nav) {
        nav.classList.toggle('scrolled', window.scrollY > 10);
        window.addEventListener('scroll', () => {
            nav.classList.toggle('scrolled', window.scrollY > 10);
        });
    }
}

/* Legacy support — sets a greeting text inside any element by ID.*/
function displayUserGreeting(elementId) {
    const userName = localStorage.getItem('userName');
    if (userName) {
        const element = document.getElementById(elementId);
        if (element) element.textContent = 'Welcome, ' + userName + '!';
    }
}

/* Toast Notifications */

/**
 * Shows a toast notification at the bottom of the screen.
 *
 * @param {string} message  - Text to display
 * @param {string} type     - 'success' | 'error' | '' (dark default)
 * @param {number} duration - How long to show in ms (default 3000)
 * @param {string} toastId  - ID of the toast element (default 'mmToast')
 */
function showToast(message, type = '', duration = 3000, toastId = 'mmToast') {
    const toast = document.getElementById(toastId);
    if (!toast) return;
    toast.textContent = message;
    toast.className   = 'mm-toast ' + type + ' show';
    clearTimeout(toast._hideTimer);
    toast._hideTimer = setTimeout(() => {
        toast.className = 'mm-toast ' + type;
    }, duration);
}

/* User Profile */

/* Returns the parsed user profile from localStorage.*/
function getUserProfile() {
    return JSON.parse(localStorage.getItem('userProfile') || '{}');
}

/* Saves a profile object to localStorage.*/
function setUserProfile(profile) {
    localStorage.setItem('userProfile', JSON.stringify(profile));
}

/* Profile Completeness */

function updateCompleteness() {
    const name    = localStorage.getItem('userName') || '';
    const profile = JSON.parse(localStorage.getItem('userProfile') || 'null');
    const degree  = profile?.degree && profile.degree !== 'Not specified';
    const skills  = (profile?.skills || []).length >= 3;
    const hasCV   = !!profile;

    const checks = [
        { id: 'checkPersonal', pass: !!name },
        { id: 'checkDegree',   pass: !!degree },
        { id: 'checkSkills',   pass: skills },
        { id: 'checkCV',       pass: hasCV },
    ];

    let score = 0;
    checks.forEach(c => {
        const el = document.getElementById(c.id);
        if (!el) return;
        if (c.pass) {
            el.textContent = '✓';
            el.style.color = '#059669';
            score += 25;
        } else {
            el.textContent = '—';
            el.style.color = 'var(--muted)';
        }
    });

    const bar   = document.getElementById('completenessBar');
    const label = document.getElementById('completenessLabel');
    if (label) label.textContent = score + '%';
    if (bar) setTimeout(() => { bar.style.width = score + '%'; }, 200);
}

/* Saved Jobs */

/* Returns array of saved jobs from localStorage.*/
function getSavedJobs() {
    return JSON.parse(localStorage.getItem('savedJobs') || '[]');
}

/* Saves a job to saved jobs list. Returns true if added, false if already exists.*/
function saveJob(job) {
    const saved = getSavedJobs();
    if (saved.some(j => j.job_id === job.job_id)) return false;
    saved.push(job);
    localStorage.setItem('savedJobs', JSON.stringify(saved));
    return true;
}

/* Removes a job from saved jobs by job_id.*/
function unsaveJob(jobId) {
    const saved   = getSavedJobs();
    const updated = saved.filter(j => j.job_id !== jobId);
    localStorage.setItem('savedJobs', JSON.stringify(updated));
}

/* Returns true if a job is currently saved.*/
function isJobSaved(jobId) {
    return getSavedJobs().some(j => j.job_id === jobId);
}

/* Formatting Helpers */
/* Formats bytes into human-readable file size string.*/
function formatFileSize(bytes) {
    if (bytes < 1024)        return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/* Formats a salary number as a GBP string: e.g. £45,000 */
function formatSalary(amount) {
    if (!amount) return '—';
    return '£' + Number(amount).toLocaleString('en-GB');
}

/* Formats min and max salary as a range string: e.g. £35,000 – £50,000*/
function formatSalaryRange(min, max) {
    if (!min && !max) return '—';
    if (!max) return formatSalary(min);
    return formatSalary(min) + ' – ' + formatSalary(max);
}

/* Returns a score class string based on match percentage. */
function getScoreClass(score) {
    if (score >= 80) return 'score-high';
    if (score >= 60) return 'score-medium';
    return 'score-low';
}

/* Normalises degree text for comparison.*/
function normalizeDegreeText(s) {
    if (!s) return '';
    return s.toLowerCase().trim().replace(/\s+/g, ' ');
}

function parseSkills(skillsRaw) {
    if (!skillsRaw) return [];

    if (Array.isArray(skillsRaw)) {
        return skillsRaw
            .map(s => String(s).trim())
            .filter(Boolean);
    }

    return String(skillsRaw)
        .replace(/[\[\]']/g, '')
        .replace(/"/g, '')
        .split(',')
        .map(s => s.trim())
        .filter(Boolean);
}

/* Sleep Utility */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}