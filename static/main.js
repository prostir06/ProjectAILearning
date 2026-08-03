/**
 * Клієнтська валідація форми передбачення діабету.
 *
 * StandardJS:
 * - 'use strict'
 * - const / let (без var)
 * - без крапок з комою
 * - IIFE, щоб не засмічувати global scope
 *
 * Діапазони синхронізовані з config.VALID_RANGES (сервер — джерело правди).
 */
'use strict'

;(function () {
  // Правила для number-полів: min / max / людський підпис помилки.
  const FIELD_RULES = {
    age: { min: 1, max: 120, label: 'Вік' },
    bmi: { min: 10, max: 80, label: 'ІМТ' },
    HbA1c_level: { min: 3, max: 15, label: 'HbA1c' },
    blood_glucose_level: { min: 50, max: 400, label: 'Глюкоза в крові' }
  }

  /**
   * Валідує одне числове поле.
   *
   * @param {HTMLInputElement} input Поле форми.
   * @param {{min: number, max: number, label: string}} rule Правило.
   * @returns {string|null} Текст помилки або null.
   */
  function validateNumberField (input, rule) {
    // Захист від виклику без DOM-елемента / правила.
    if (!input || !rule) {
      return null
    }

    const raw = String(input.value).trim()
    const value = Number(raw)

    // Порожнє або нечислове значення.
    if (raw === '' || Number.isNaN(value)) {
      return rule.label + ' має бути числом.'
    }

    // Поза дозволеним діапазоном.
    if (value < rule.min || value > rule.max) {
      return rule.label + ' має бути в діапазоні ' + rule.min + '–' + rule.max + '.'
    }

    return null
  }

  /**
   * Показує або прибирає aria-повідомлення про помилку біля поля.
   *
   * @param {HTMLInputElement} input Поле форми.
   * @param {string|null} message Текст помилки.
   */
  function setFieldError (input, message) {
    if (!input || !input.id) {
      return
    }

    try {
      const errorId = input.id + '-error'
      let existing = document.getElementById(errorId)

      if (message) {
        // Позначаємо поле невалідним для скрінрідерів.
        input.setAttribute('aria-invalid', 'true')
        if (!existing) {
          existing = document.createElement('span')
          existing.id = errorId
          existing.className = 'field-error'
          existing.setAttribute('role', 'alert')
          input.insertAdjacentElement('afterend', existing)
        }
        existing.textContent = message
        return
      }

      // Очищення попередньої помилки.
      input.removeAttribute('aria-invalid')
      if (existing) {
        existing.remove()
      }
    } catch (error) {
      // У тестах / обмежених середовищах DOM може бути недоступний.
      console.warn('Не вдалося оновити помилку поля:', error)
    }
  }

  /**
   * Валідує всі числові поля форми перед submit.
   *
   * @param {HTMLFormElement} form Форма передбачення.
   * @returns {boolean} true, якщо все валідно.
   */
  function validateForm (form) {
    if (!form || !form.elements) {
      return false
    }

    let isValid = true

    Object.keys(FIELD_RULES).forEach(function (fieldName) {
      const input = form.elements.namedItem(fieldName)
      if (!(input instanceof HTMLInputElement)) {
        return
      }

      const errorMessage = validateNumberField(input, FIELD_RULES[fieldName])
      setFieldError(input, errorMessage)
      if (errorMessage) {
        isValid = false
      }
    })

    return isValid
  }

  /**
   * Синхронізує слайдер порогу з текстовим виводом і ARIA.
   *
   * @param {HTMLInputElement} slider input[type=range].
   * @param {HTMLElement} display Елемент <output> / span.
   */
  function updateThresholdDisplay (slider, display) {
    if (!(slider instanceof HTMLInputElement) || !display) {
      return
    }

    try {
      const value = slider.value
      display.textContent = value + '%'
      slider.setAttribute('aria-valuenow', value)
      slider.setAttribute('aria-valuetext', value + ' відсотків')
    } catch (error) {
      console.warn('Не вдалося оновити поріг:', error)
    }
  }

  // Ініціалізація після побудови DOM (скрипт підключено з defer).
  document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('prediction-form')
    if (!form) {
      return
    }

    // Блокуємо submit, якщо клієнтська валідація не пройшла.
    form.addEventListener('submit', function (event) {
      if (!validateForm(form)) {
        event.preventDefault()
      }
    })

    // Жива перевірка під час введення.
    Object.keys(FIELD_RULES).forEach(function (fieldName) {
      const input = form.elements.namedItem(fieldName)
      if (input instanceof HTMLInputElement) {
        input.addEventListener('input', function () {
          setFieldError(
            input,
            validateNumberField(input, FIELD_RULES[fieldName])
          )
        })
      }
    })

    // Слайдер порогу класифікації «Так» / «Ні».
    const thresholdSlider = document.getElementById('prediction_threshold')
    const thresholdDisplay = document.getElementById('threshold-display')
    if (thresholdSlider instanceof HTMLInputElement && thresholdDisplay) {
      thresholdSlider.addEventListener('input', function () {
        updateThresholdDisplay(thresholdSlider, thresholdDisplay)
      })
      updateThresholdDisplay(thresholdSlider, thresholdDisplay)
    }
  })
}())
