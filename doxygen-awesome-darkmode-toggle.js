/**
 * Doxygen Awesome Dark Mode Toggle
 * https://github.com/jothepro/doxygen-awesome-css
 */
class DoxygenAwesomeDarkModeToggle extends HTMLElement {
    static init() {
        $(function() {
            $(document).ready(function() {
                const toggleButton = document.createElement('doxygen-awesome-dark-mode-toggle');
                toggleButton.title = "Toggle Light/Dark Theme";
                
                const searchBox = document.getElementById("MSearchBox");
                if (searchBox) {
                    searchBox.parentNode.appendChild(toggleButton);
                } else {
                    const top = document.getElementById("top");
                    if (top) top.appendChild(toggleButton);
                }
            });
        });
    }

    constructor() {
        super();
        this.onclick = this.toggleDarkMode;
    }

    static get userPreference() {
        return localStorage.getItem('theme');
    }

    static set userPreference(preference) {
        if (preference) {
            localStorage.setItem('theme', preference);
        } else {
            localStorage.removeItem('theme');
        }
    }

    static get systemPreference() {
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    static get isDarkMode() {
        const pref = DoxygenAwesomeDarkModeToggle.userPreference;
        return pref ? pref === 'dark' : DoxygenAwesomeDarkModeToggle.systemPreference === 'dark';
    }

    static enableDarkMode(enable) {
        if (enable) {
            document.documentElement.classList.add("dark-mode");
            document.documentElement.classList.remove("light-mode");
        } else {
            document.documentElement.classList.add("light-mode");
            document.documentElement.classList.remove("dark-mode");
        }
    }

    connectedCallback() {
        this.updateIcon();
        DoxygenAwesomeDarkModeToggle.enableDarkMode(DoxygenAwesomeDarkModeToggle.isDarkMode);
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
            if (!DoxygenAwesomeDarkModeToggle.userPreference) {
                DoxygenAwesomeDarkModeToggle.enableDarkMode(event.matches);
                this.updateIcon();
            }
        });
    }

    toggleDarkMode() {
        const isDark = !DoxygenAwesomeDarkModeToggle.isDarkMode;
        DoxygenAwesomeDarkModeToggle.userPreference = isDark ? 'dark' : 'light';
        DoxygenAwesomeDarkModeToggle.enableDarkMode(isDark);
        this.updateIcon();
    }

    updateIcon() {
        const isDark = DoxygenAwesomeDarkModeToggle.isDarkMode;
        this.innerHTML = isDark 
            ? `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`
            : `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;
    }
}

customElements.define("doxygen-awesome-dark-mode-toggle", DoxygenAwesomeDarkModeToggle);
DoxygenAwesomeDarkModeToggle.init();
